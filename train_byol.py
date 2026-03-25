"""
  developed by Yi-Jiun Su
  The latest update in April 2024
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
from utils.pixcl_multi import BYOLTrainer, MLP, NetWrapperMultiLayers
from utils.runtime import (
    configure_torch_runtime,
    get_best_device,
    warn_if_apple_silicon_mps_unavailable,
)


def normalize_optional(value):
    if value in (None, "", "None", "none", "null", "Null"):
        return None
    return value


def build_optimizer(config, online_encoder, online_projector, online_predictor):
    parameters = (
        list(online_encoder.parameters())
        + list(online_projector.parameters())
        + list(online_predictor.parameters())
    )
    if config["opt_method"] == "adam_hd":
        raise ValueError(
            "opt_method='adam_hd' is not supported in this repository because "
            "the Adam_HD implementation is not bundled. Use 'adam' or 'radam'."
        )
    if config["opt_method"] == "radam":
        return torch.optim.RAdam(parameters, **config["optimizer"])
    return torch.optim.Adam(parameters, **config["optimizer"])


def main():
    distributed = init_distributed_mode(get_best_device())
    try:
        config = yaml.safe_load(open("config_byol.yaml", "r"))
        device = distributed.device
        configure_torch_runtime(device)
        warn_if_apple_silicon_mps_unavailable(device)
        if distributed.is_main_process:
            print(torch.__version__)
            print(f"Learning with: {device}")
            if distributed.enabled:
                print(f"Distributed BYOL with world size {distributed.world_size}")

        pretrain_dir = normalize_optional(config.get("pretrain_dir"))
        pretrain_epoch = normalize_optional(config.get("pretrain_epoch"))

        # define data Transform functions
        # T1 = T.Compose([CT.Jitter(),CT.Scale()])
        augment1 = T.Compose([CT.Jitter(), CT.FFT()])
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
        # The classifier head is never used by the SSL objective. Removing it
        # avoids DDP tracking permanently-unused parameters from the ResNet fc.
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
            instance_representation, _ = online_encoder(sample)

        online_projector = MLP(
            instance_representation.shape[1],
            config["learner"]["projection_size"],
            config["learner"]["projection_hidden_size"],
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
                if "online_projector_state_dict" in load_params:
                    online_projector.load_state_dict(load_params["online_projector_state_dict"])
                if "online_predictor_state_dict" in load_params:
                    online_predictor.load_state_dict(load_params["online_predictor_state_dict"])
            except FileNotFoundError:
                if distributed.is_main_process:
                    print("Pre-trained weights not found. Training from scratch.")

        with torch.no_grad():
            target_encoder = deepcopy(online_encoder)
            for parameter in target_encoder.parameters():
                parameter.requires_grad = False
            target_projector = deepcopy(online_projector)
            for parameter in target_projector.parameters():
                parameter.requires_grad = False

        if distributed.enabled:
            online_encoder = wrap_ddp(online_encoder, distributed)
            online_projector = wrap_ddp(online_projector, distributed)
            online_predictor = wrap_ddp(online_predictor, distributed)

        optimizer = build_optimizer(config, online_encoder, online_projector, online_predictor)
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

        learner = BYOLTrainer(
            online_encoder=online_encoder,
            target_encoder=target_encoder,
            online_projector=online_projector,
            target_projector=target_projector,
            online_predictor=online_predictor,
            optimizer=optimizer,
            scheduler=scheduler,
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
