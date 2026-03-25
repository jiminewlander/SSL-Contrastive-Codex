"""
  developed by Yi-Jiun Su
  the latest update in April 2024
"""
import os
from copy import deepcopy

import torch
import torchvision
import torchvision.transforms as T
import yaml

import utils.custom_transform as CT
from utils.data_loader import CustomDataset
from utils.distributed import cleanup_distributed, init_distributed_mode, wrap_ddp
from utils.pixcl_multi import ConvMLP, MLP, NetWrapperMultiLayers, PPM, PixclLearner
from utils.runtime import (
    configure_torch_runtime,
    get_best_device,
    warn_if_apple_silicon_mps_unavailable,
)


def normalize_optional(value):
    if value in (None, "", "None", "none", "null", "Null"):
        return None
    return value


def build_optimizer(
    config,
    online_encoder,
    online_instance_projector,
    online_pixel_projector,
    propagate_pixels,
    online_predictor,
):
    parameters = (
        list(online_encoder.parameters())
        + list(online_instance_projector.parameters())
        + list(online_pixel_projector.parameters())
        + list(propagate_pixels.parameters())
        + list(online_predictor.parameters())
    )
    if config["opt_method"] == "radam":
        return torch.optim.RAdam(parameters, **config["optimizer"])
    return torch.optim.Adam(parameters, **config["optimizer"])


def main():
    distributed = init_distributed_mode(get_best_device())
    try:
        config = yaml.safe_load(open("config_pixcl.yaml", "r"))
        device = distributed.device
        configure_torch_runtime(device)
        warn_if_apple_silicon_mps_unavailable(device)
        if distributed.is_main_process:
            print(torch.__version__)
            print(f"Learning with: {device}")
            if distributed.enabled:
                print(f"Distributed PIXCL with world size {distributed.world_size}")

        pretrain_dir = normalize_optional(config.get("pretrain_dir"))
        pretrain_epoch = normalize_optional(config.get("pretrain_epoch"))

        augment1 = T.Compose([CT.Jitter(), CT.Scale()])
        augment2 = T.Compose([CT.WrapHW(dim=2), CT.WrapHW(dim=3)])
        dataset = CustomDataset(**config["dataset"])

        if config["backbone_arch"] == "resnet18":
            resnet = torchvision.models.resnet18()
        elif config["backbone_arch"] == "resnet34":
            resnet = torchvision.models.resnet34()
        elif config["backbone_arch"] == "resnet50":
            resnet = torchvision.models.resnet50()
        else:
            raise ValueError(f"Unsupported backbone_arch: {config['backbone_arch']}")
        # The classifier head is unused for SSL pretraining and only causes
        # DDP to track parameters that never receive gradients.
        resnet.fc = torch.nn.Identity()

        online_encoder = NetWrapperMultiLayers(net=resnet).to(device)

        with torch.no_grad():
            sample = torch.randn(
                1,
                config["dataset"]["channels"],
                config["learner"]["image_sizeH"],
                config["learner"]["image_sizeW"],
                device=device,
            )
            instance_representation, pixel_representation = online_encoder(sample)

        online_instance_projector = MLP(
            instance_representation.shape[1],
            config["learner"]["projection_size"],
            config["learner"]["projection_hidden_size"],
        ).to(device)
        online_pixel_projector = ConvMLP(
            pixel_representation.shape[1],
            config["learner"]["projection_size"],
            config["learner"]["projection_hidden_size"],
        ).to(device)
        propagate_pixels = PPM(
            chan=config["learner"]["projection_size"], **config["ppm"]
        ).to(device)
        online_predictor = MLP(
            config["learner"]["projection_size"],
            config["learner"]["projection_size"],
            config["learner"]["projection_hidden_size"],
        ).to(device)

        load_params = None
        if pretrain_dir:
            try:
                checkpoints_dir = os.path.join("./runs", pretrain_dir, "checkpoints")
                load_params = torch.load(
                    os.path.join(checkpoints_dir, "model_epoch" + pretrain_epoch + ".pth"),
                    map_location=torch.device(device),
                )
                online_encoder.load_state_dict(load_params["online_encoder_state_dict"])
                if "online_instance_projector_state_dict" in load_params:
                    online_instance_projector.load_state_dict(
                        load_params["online_instance_projector_state_dict"]
                    )
                if "online_pixel_projector_state_dict" in load_params:
                    online_pixel_projector.load_state_dict(
                        load_params["online_pixel_projector_state_dict"]
                    )
                if "online_predictor_state_dict" in load_params:
                    online_predictor.load_state_dict(
                        load_params["online_predictor_state_dict"]
                    )
                if "propagate_pixels_state_dict" in load_params:
                    propagate_pixels.load_state_dict(
                        load_params["propagate_pixels_state_dict"]
                    )
            except FileNotFoundError:
                if distributed.is_main_process:
                    print("Pre-trained weights not found. Training from scratch.")

        with torch.no_grad():
            target_encoder = deepcopy(online_encoder)
            for parameter in target_encoder.parameters():
                parameter.requires_grad = False
            target_instance_projector = deepcopy(online_instance_projector)
            for parameter in target_instance_projector.parameters():
                parameter.requires_grad = False
            target_pixel_projector = deepcopy(online_pixel_projector)
            for parameter in target_pixel_projector.parameters():
                parameter.requires_grad = False

        if distributed.enabled:
            online_encoder = wrap_ddp(online_encoder, distributed)
            online_instance_projector = wrap_ddp(online_instance_projector, distributed)
            online_pixel_projector = wrap_ddp(online_pixel_projector, distributed)
            propagate_pixels = wrap_ddp(propagate_pixels, distributed)
            online_predictor = wrap_ddp(online_predictor, distributed)

        optimizer = build_optimizer(
            config,
            online_encoder,
            online_instance_projector,
            online_pixel_projector,
            propagate_pixels,
            online_predictor,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **config["lr_sch_param"]
        )

        if load_params and "optimizer_state_dict" in load_params:
            try:
                optimizer.load_state_dict(load_params["optimizer_state_dict"])
            except ValueError:
                if distributed.is_main_process:
                    print(
                        "Optimizer state incompatible with current model. "
                        "Continuing with a fresh optimizer."
                    )

        learner = PixclLearner(
            online_encoder=online_encoder,
            target_encoder=target_encoder,
            online_instance_projector=online_instance_projector,
            target_instance_projector=target_instance_projector,
            online_pixel_projector=online_pixel_projector,
            target_pixel_projector=target_pixel_projector,
            optimizer=optimizer,
            scheduler=scheduler,
            propagate_pixels=propagate_pixels,
            online_predictor=online_predictor,
            device=device,
            augment1=augment1,
            augment2=augment2,
            distributed=distributed,
            **config["learner"],
        )
        learner.train(dataset)
    finally:
        cleanup_distributed(distributed)


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
