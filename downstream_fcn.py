"""
  developed by Yi-Jiun Su
  the latest update in March 2024
"""
from datetime import datetime
import logging
import os
from shutil import copyfile

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision
import yaml
from kornia.losses import BinaryFocalLossWithLogits

from utils.data_loader_downstream import CustomDataset
from utils.dice_score import focal_tversky_loss
from utils.distributed import (
    cleanup_distributed,
    init_distributed_mode,
    reduce_mean,
    reduce_sum,
    unwrap_module,
    wrap_ddp,
)
from utils.fcn import FCN16s, FCN32s, FCN8s, FCNs
from utils.hausdorff import HausdorffERLoss, hausdorff_distance
from utils.pixcl_multi import NetWrapperMultiLayers
from utils.runtime import (
    autocast_context,
    configure_torch_runtime,
    get_best_device,
    grad_scaler,
    resolve_num_workers,
    should_pin_memory,
    warn_if_apple_silicon_mps_unavailable,
)


def normalize_optional(value):
    if value in (None, "", "None", "none", "null", "Null"):
        return None
    return value


def _create_model_training_folder(writer, files_to_same):
    model_checkpoints_folder = os.path.join(writer.log_dir, "checkpoints")
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        for file in files_to_same:
            copyfile(file, os.path.join(model_checkpoints_folder, os.path.basename(file)))


@torch.inference_mode()
def evaluate(net, dataloader, device, criterion, distributed):
    net.eval()
    running_loss_f = 0.0
    running_loss_t = 0.0
    running_loss_h = 0.0
    local_batches = 0

    iterator = dataloader
    if distributed.is_main_process:
        iterator = tqdm(
            dataloader,
            total=len(dataloader),
            desc="Validation",
            unit="batch",
            leave=False,
        )

    with autocast_context(device):
        for batch in iterator:
            image, true_mask = batch["image"], batch["mask"]
            image = image.to(device=device, dtype=torch.float32)
            true_mask = true_mask.to(device=device, dtype=torch.long)
            pred_mask = net(image)
            assert true_mask.min() >= 0 and true_mask.max() <= 1, "True mask indices should be in [0, 1]"
            loss1 = criterion(pred_mask.squeeze(1), true_mask.float())
            loss2 = focal_tversky_loss(
                torch.sigmoid(pred_mask.squeeze(1)),
                true_mask.float(),
                multiclass=False,
                alpha=0.3,
                beta=0.7,
                gamma=2.0,
            )
            mask = torch.sigmoid(pred_mask) > 0.5
            batch_hd = []
            for pred_item, true_item in zip(mask[:, 0], true_mask):
                hd = hausdorff_distance(pred_item.cpu().numpy(), true_item.cpu().numpy())
                batch_hd.append(hd["mean"])
            running_loss_f += loss1.item()
            running_loss_t += loss2.item()
            running_loss_h += sum(batch_hd) / max(len(batch_hd), 1)
            local_batches += 1

    stats = torch.tensor(
        [running_loss_f, running_loss_t, running_loss_h, local_batches],
        device=device,
        dtype=torch.float64,
    )
    stats = reduce_sum(stats, distributed)
    total_batches = max(int(stats[3].item()), 1)
    net.train()
    return (
        (stats[0] / total_batches).item(),
        (stats[1] / total_batches).item(),
        (stats[2] / total_batches).item(),
    )


def build_fcn(config, backbone, in_channels):
    if config["downstream_arch"] == "fcn32s":
        return FCN32s(
            pretrained_net=backbone,
            n_class=config["image"]["n_class"],
            in_channels=in_channels,
        )
    if config["downstream_arch"] == "fcn16s":
        return FCN16s(
            pretrained_net=backbone,
            n_class=config["image"]["n_class"],
            in_channels=in_channels,
        )
    if config["downstream_arch"] == "fcn8s":
        return FCN8s(
            pretrained_net=backbone,
            n_class=config["image"]["n_class"],
            in_channels=in_channels,
        )
    return FCNs(
        pretrained_net=backbone,
        n_class=config["image"]["n_class"],
        in_channels=in_channels,
    )


def shard_validation_set(val_set, distributed):
    if not distributed.enabled:
        return val_set
    indices = torch.arange(len(val_set))
    shards = torch.tensor_split(indices, distributed.world_size)
    return Subset(val_set, shards[distributed.rank].tolist())


def main():
    distributed = init_distributed_mode(get_best_device())
    try:
        config = yaml.safe_load(open("config_fcn.yaml", "r"))
        device = distributed.device
        configure_torch_runtime(device)
        warn_if_apple_silicon_mps_unavailable(device)
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
        if distributed.is_main_process:
            print(f"Training with: {device}")
            if distributed.enabled:
                print(f"Distributed FCN with world size {distributed.world_size}")

        pretrain_dir = normalize_optional(config.get("pretrain_dir"))
        pretrain_epoch = normalize_optional(config.get("pretrain_epoch"))

        if config["backbone_arch"] == "resnet18":
            resnet = torchvision.models.resnet18()
        elif config["backbone_arch"] == "resnet34":
            resnet = torchvision.models.resnet34()
        elif config["backbone_arch"] == "resnet50":
            resnet = torchvision.models.resnet50()
        else:
            raise ValueError(f"Unsupported backbone_arch: {config['backbone_arch']}")

        backbone = NetWrapperMultiLayers(net=resnet).to(device)
        with torch.no_grad():
            sample = torch.randn(
                1,
                config["dataset"]["channels"],
                config["image"]["sizeH"],
                config["image"]["sizeW"],
                device=device,
            )
            _, c5 = backbone(sample)

        if pretrain_dir:
            try:
                checkpoints_dir = os.path.join(pretrain_dir, "checkpoints")
                load_params = torch.load(
                    os.path.join(checkpoints_dir, "model_epoch" + pretrain_epoch + ".pth"),
                    map_location=torch.device(device),
                )
                backbone.load_state_dict(load_params["online_encoder_state_dict"])
            except FileNotFoundError:
                if distributed.is_main_process:
                    print("Pre-trained weights not found. Training from scratch.")

        fcn = build_fcn(config, backbone, c5.shape[1]).to(device)
        if distributed.enabled:
            fcn = wrap_ddp(fcn, distributed)

        dataset = CustomDataset(**config["dataset"])
        n_val = int(len(dataset) * config["trainer"]["val_percent"])
        n_train = len(dataset) - n_val
        train_set, val_set = random_split(
            dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(0),
        )
        val_set = shard_validation_set(val_set, distributed)

        train_sampler = None
        if distributed.enabled:
            train_sampler = DistributedSampler(
                train_set,
                num_replicas=distributed.world_size,
                rank=distributed.rank,
                shuffle=True,
                drop_last=False,
            )

        loader_args = dict(
            batch_size=config["trainer"]["batch_size"],
            num_workers=resolve_num_workers(config["trainer"]["num_workers"]),
            pin_memory=should_pin_memory(device),
        )
        train_loader = DataLoader(
            train_set,
            shuffle=train_sampler is None,
            drop_last=False,
            sampler=train_sampler,
            **loader_args,
        )
        val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)

        kwargs = {"alpha": 0.6, "gamma": 2.0, "reduction": "mean"}
        criterion = BinaryFocalLossWithLogits(**kwargs)
        criterion_h = HausdorffERLoss(alpha=2.0, erosions=10)
        optimizer = torch.optim.Adam(fcn.parameters(), **config["optimizer"])

        writer = None
        model_checkpoints_folder = None
        if distributed.is_main_process:
            writer = SummaryWriter(log_dir=os.path.join("runs", "fcn_" + datetime.now().strftime("%Y%m%d-%H%M%S")))
            _create_model_training_folder(writer, files_to_same=["./config_fcn.yaml", "./downstream_fcn.py"])
            model_checkpoints_folder = os.path.join(writer.log_dir, "checkpoints")

        scaler = grad_scaler(device)
        global_step = 0
        max_epochs = config["trainer"]["max_epochs"]

        for epoch_counter in range(1, max_epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch_counter)

            fcn.train()
            epoch_loss = torch.tensor(0.0, device=device)
            local_train_total = len(train_sampler) if train_sampler is not None else n_train
            progress = tqdm(
                total=local_train_total,
                desc=f"Epoch {epoch_counter}/{max_epochs}",
                unit="img",
                disable=not distributed.is_main_process,
            )

            with progress as pbar:
                for batch in train_loader:
                    images, true_masks = batch["image"], batch["mask"]
                    images = images.to(device=device, dtype=torch.float32)
                    true_masks = true_masks.to(device=device, dtype=torch.long)

                    with autocast_context(device):
                        pred_masks = fcn(images)
                        loss1 = criterion(pred_masks.squeeze(1), true_masks.float())
                        loss2 = focal_tversky_loss(
                            torch.sigmoid(pred_masks.squeeze(1)),
                            true_masks.float(),
                            multiclass=False,
                            alpha=0.3,
                            beta=0.7,
                            gamma=2.0,
                        )
                        loss3 = criterion_h(
                            F.log_softmax(pred_masks, dim=1),
                            true_masks[:, None, :, :],
                        )
                        lamda_f = loss2.item() / max(loss1.item(), 1e-6)
                        lamda_h = loss2.item() / max(loss3.item(), 1e-6)
                        loss = lamda_f * loss1 + loss2 + lamda_h * loss3

                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(fcn.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()

                    global_step += 1
                    epoch_loss += loss.detach()
                    pbar.update(images.shape[0])
                    if distributed.is_main_process:
                        pbar.set_postfix(**{"batch loss": loss1.item() + loss2.item()})
                    if writer is not None:
                        writer.add_scalar("total loss", loss.item(), global_step=global_step)
                        writer.add_scalar("FCE loss", loss1.item(), global_step=global_step)
                        writer.add_scalar("FT loss", loss2.item(), global_step=global_step)
                        writer.add_scalar("HD loss", loss3.item(), global_step=global_step)

                loss_fce, loss_ft, loss_hd = evaluate(fcn, val_loader, device, criterion, distributed)
                if distributed.is_main_process:
                    logging.info(
                        "Epoch %s Validation loss: %8.5f %8.5f %11.5f",
                        epoch_counter,
                        loss_fce,
                        loss_ft,
                        loss_hd,
                    )
                    if writer is not None:
                        writer.add_scalar("val FCE loss", loss_fce, global_step=epoch_counter)
                        writer.add_scalar("val FT loss", loss_ft, global_step=epoch_counter)
                        writer.add_scalar("val HD loss", loss_hd, global_step=epoch_counter)

            epoch_loss /= max(len(train_loader), 1)
            epoch_loss_reduced = reduce_mean(epoch_loss, distributed)
            if writer is not None:
                writer.add_scalar("epoch loss", epoch_loss_reduced.item(), global_step=epoch_counter)
                torch.save(
                    {
                        "fcn_state_dict": unwrap_module(fcn).state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    os.path.join(model_checkpoints_folder, "model_epoch" + str(epoch_counter) + ".pth"),
                )
    finally:
        cleanup_distributed(distributed)


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
