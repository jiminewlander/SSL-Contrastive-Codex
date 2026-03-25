"""
  consolidated and modified from two sources by Yi-Jiun Su 
     https://github.com/lucidrains/byol-pytorch/blob/master/byol_pytorch/byol_pytorch.py
     https://github.com/lucidrains/pixel-level-contrastive-learning/blob/main/pixel_level_contrastive_learning/pixel_level_contrastive_learning.py

  The latest update in April 2024 by Y.-J. Su
"""
import torch
from torch import nn, einsum
import torch.nn.functional as F
from math import sqrt, floor
from copy import deepcopy
import random
from einops import rearrange
import os
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from datetime import datetime
from tqdm import tqdm
import torchvision
from utils.distributed import reduce_mean, unwrap_module
from utils.runtime import resolve_num_workers, should_pin_memory

def _create_model_training_folder(writer, files_to_same):
    model_checkpoints_folder = os.path.join(writer.log_dir, 'checkpoints')
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        for file in files_to_same:
            copyfile(file, os.path.join(model_checkpoints_folder, os.path.basename(file)))

def cutout_coordinateW(image, ratio_range = (0.7, 0.9)):
    _, _, h, orig_w = image.shape
    ratio_lo, ratio_hi = ratio_range
    random_ratio = ratio_lo + random.random() * (ratio_hi - ratio_lo)
    w = floor(random_ratio * orig_w)
    coor_x = floor((orig_w - w) *random.random())
    return ((0,h), (coor_x, coor_x+w)), random_ratio

def cutout_and_resize(image, coordinates, output_size = None, mode = 'nearest'):
    output_size = image.shape[2:] if output_size is None else output_size
    (y0, y1), (x0, x1) = coordinates
    cutout_image = image[:, :, y0:y1, x0:x1]
    return F.interpolate(cutout_image, size = output_size, mode = mode)

# exponential moving average
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

# loss fn defined as mean squared error in Grill et al. [2020]
def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

# pairwise_angle considered as part of the loss function 
def pairwise_angle(x, y):
    dotprod = torch.einsum('ij,ij->i', x, y)
    x = torch.linalg.norm(x,axis=1)
    y = torch.linalg.norm(y,axis=1)
    return torch.arccos(dotprod/x/y)

# Multi-Layer Perceptorn for instance-level projector
class MLP(nn.Module):
    def __init__(self, chan, chan_out = 256, inner_dim = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(chan, inner_dim),
            nn.BatchNorm1d(inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, chan_out)
        )
    def forward(self, x):
        return self.net(x)

# Pixel-to-Propagation Module (PPM) - the last step of online network of PixPro
class PPM(nn.Module):
    def __init__(
        self,
        *,
        chan,
        num_layers = 1,
        gamma = 2):
        super().__init__()
        self.gamma = gamma
        if num_layers == 0:
            self.transform_net = nn.Identity()
        elif num_layers == 1:
            self.transform_net = nn.Conv2d(chan, chan, 1)
        elif num_layers == 2:
            self.transform_net = nn.Sequential(
                nn.Conv2d(chan, chan, 1),
                nn.BatchNorm2d(chan),
                nn.ReLU(),
                nn.Conv2d(chan, chan, 1)
            )
        else:
            raise ValueError('num_layers must be one of 0, 1, or 2')
    def forward(self, x):
        xi = x[:, :, :, :, None, None]
        xj = x[:, :, None, None, :, :]
        similarity = F.relu(F.cosine_similarity(xi, xj, dim = 1)) ** self.gamma
        transform_out = self.transform_net(x)
        out = einsum('b x y h w, b c h w -> b c x y', similarity, transform_out)
        return out

# Multi-Layer Perceptron for pixel-level projector
class ConvMLP(nn.Module):
    def __init__(self, chan, chan_out = 256, inner_dim = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(chan, inner_dim, 1),
            nn.BatchNorm2d(inner_dim),
            nn.ReLU(),
            nn.Conv2d(inner_dim, chan_out, 1)
        )
    def forward(self, x):
        return self.net(x)

# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projecter and predictor nets
# ResNet Layer_ID
#   0 -10'conv1' - Conv2d
#   1 -9 'bn1' - BatchNorm2d
#   2 -8 'relu' - ReLu - c0
#   3 -7 'maxpool' - MaxPool2d - c1
#   4 -6 'layer1' - c2
#   5 -5 'layer2' - c3
#   6 -4 'layer3' - c4
#   7 -3 'layer4' - pixel_layer
#   8 -2 'avgpool - instance_layer
#   9 -1 'fc'
class NetWrapperMultiLayers(nn.Module):
    def __init__(
        self, 
        *,
        net, 
        layer_IDs = [2,3,4,5,6,7,8]
    ):
        super().__init__()
        self.net = net
        self.layer_IDs = layer_IDs
        self.num_layers = len(layer_IDs)
        
        self.hook_registered = False 
        self.hidden_c0 = None
        self.hidden_c1 = None
        self.hidden_c2 = None
        self.hidden_c3 = None
        self.hidden_c4 = None
        self.hidden_pixel = None
        self.hidden_instance = None
    """
       layer_ids can be numbers or strings
    """
    def _find_layer(self, layer_id):
        if type(layer_id) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(layer_id, None)
        elif type(layer_id) == int:
            children = [*self.net.children()]
            return children[layer_id]
        return None

    # assigned label to the output for specific hidden layer
    def _hook_c0(self, _, __, output):
        setattr(self, 'hidden_c0', output)
    def _hook_c1(self, _, __, output):
        setattr(self, 'hidden_c1', output)
    def _hook_c2(self, _, __, output):
        setattr(self, 'hidden_c2', output)
    def _hook_c3(self, _, __, output):
        setattr(self, 'hidden_c3', output)
    def _hook_c4(self, _, __, output):
        setattr(self, 'hidden_c4', output)
    def _hook_pixel(self, _, __, output):
        setattr(self, 'hidden_pixel', output)
    def _hook_instance(self, _, __, output):
        setattr(self, 'hidden_instance', output)

    def _register_hook(self):
        for i in range(self.num_layers):
            layer_id = self.layer_IDs[i]
            if layer_id in [2, -8, 'relu']:
                c0_layer = self._find_layer(layer_id)
                assert c0_layer is not None, f'hidden layer ({layer_id}) not found'
                c0_layer.register_forward_hook(self._hook_c0)
            elif layer_id in [3, -7, 'maxpool']:
                c1_layer = self._find_layer(layer_id)
                assert c1_layer is not None, f'hidden layer ({layer_id}) not found'
                c1_layer.register_forward_hook(self._hook_c1)
            elif layer_id in [4, -6, 'layer1']:
                c2_layer = self._find_layer(layer_id)
                assert c2_layer is not None, f'hidden layer ({layer_id}) not found'
                c2_layer.register_forward_hook(self._hook_c2)
            elif layer_id in [5, -5, 'layer2']:
                c3_layer = self._find_layer(layer_id)
                assert c3_layer is not None, f'hidden layer ({layer_id}) not found'
                c3_layer.register_forward_hook(self._hook_c3)
            elif layer_id in [6, -4, 'layer3']:
                c4_layer = self._find_layer(layer_id)
                assert c4_layer is not None, f'hidden layer ({layer_id}) not found'
                c4_layer.register_forward_hook(self._hook_c4)
            elif layer_id in [7, -3, 'layer4']:
                pixel_layer = self._find_layer(layer_id)
                assert pixel_layer is not None, f'hidden layer ({layer_id}) not found'
                pixel_layer.register_forward_hook(self._hook_pixel)
            elif layer_id in [8, -2, 'avgpool']:
                instance_layer = self._find_layer(layer_id)
                assert instance_layer is not None, f'hidden layer ({layer_id}) not found'
                instance_layer.register_forward_hook(self._hook_instance)
            elif layer_id in [0, 1, 9, -10, -9, -1, 'conv1', 'bn1', 'fc']:
                print(f'hidden layer ({layer_id}) not specified for output')
            else:
                assert self._find_layer(layer_id) is not None, f'hidden layer ({layer_id}) not found'
        self.hook_registered = True

    def get_representation_multi(self, x):
        if not self.hook_registered:
            self._register_hook()
        _ = self.net(x)
        hidden_c0 = self.hidden_c0
        hidden_c1 = self.hidden_c1
        hidden_c2 = self.hidden_c2
        hidden_c3 = self.hidden_c3
        hidden_c4 = self.hidden_c4
        hidden_pixel = self.hidden_pixel
        hidden_instance = self.hidden_instance
        self.hidden_c0 = None
        self.hidden_c1 = None
        self.hidden_c2 = None
        self.hidden_c3 = None
        self.hidden_c4 = None
        self.hidden_pixel = None
        self.hidden_instance = None
        return hidden_instance, hidden_pixel, hidden_c4, hidden_c3, hidden_c2, hidden_c1, hidden_c0

    def get_representation(self, x):
        if not self.hook_registered:
            self._register_hook()
        _ = self.net(x)
        hidden_pixel = self.hidden_pixel
        hidden_instance = self.hidden_instance
        self.hidden_pixel = None
        self.hidden_instance = None
        return hidden_instance, hidden_pixel

    def forward(self, x, return_multi = False):
        if return_multi:
            return self.get_representation_multi(x)
        else:
            instance_rep, pixel_rep = self.get_representation(x)
            return instance_rep.flatten(1), pixel_rep

class BYOLTrainer():
    def __init__(
        self, 
        online_encoder,
        target_encoder, 
        online_projector,
        target_projector,
        online_predictor,
        optimizer, 
        scheduler, 
        device, 
        augment1, 
        augment2, 
        **params
    ):
        self.online_encoder = online_encoder
        self.target_encoder = target_encoder
        self.online_projector = online_projector
        self.target_projector = target_projector
        self.online_predictor = online_predictor
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.augment1 = augment1
        self.augment2 = augment2
        self.alpha = 1.
        self.image_sizeH = params['image_sizeH']
        self.image_sizeW = params['image_sizeW']
        self.projection_size = params['projection_size']
        self.projection_hidden_size = params['projection_hidden_size']
        self.target_ema_updater = EMA(params['moving_average_decay'])
        self.batch_size = params['batch_size']
        self.max_epochs = params['max_epochs']
        self.num_workers = resolve_num_workers(params['num_workers'])
        self.pdict = nn.PairwiseDistance(p=2.0)
        self.distributed = params.get('distributed')
        self.is_main_process = self.distributed is None or self.distributed.is_main_process
        self.writer = None
        if self.is_main_process:
            self.writer = SummaryWriter(log_dir=os.path.join('runs','byol_'+datetime.now().strftime("%Y%m%d-%H%M%S")))
            _create_model_training_folder(self.writer, files_to_same=["./config_byol.yaml","./train_byol.py","./utils/pixcl_multi.py"])

    def train(self, train_dataset):
        train_sampler = None
        if self.distributed is not None and self.distributed.enabled:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.distributed.world_size,
                rank=self.distributed.rank,
                shuffle=True,
                drop_last=False,
            )

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                num_workers=self.num_workers, drop_last=False, shuffle=train_sampler is None,
                sampler=train_sampler, pin_memory=should_pin_memory(self.device))
        niter = 0
        model_checkpoints_folder = None
        if self.writer is not None:
            model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        for epoch_counter in range(self.max_epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch_counter)
            lr_epoch = 0
            loss_epoch = 0
            progress = tqdm(
                train_loader,
                total=len(train_loader),
                desc=f'BYOL {epoch_counter + 1}/{self.max_epochs}',
                unit='batch',
                disable=not self.is_main_process,
                leave=False,
            )
            for batch in progress:
                image0 = batch['image']
                # Keep custom augmentations on CPU. On Apple Silicon, many small
                # tensor ops inside WrapHW are substantially slower on MPS than
                # running the transform on CPU and transferring the result once.
                image1, image2 = self.augment1(image0), self.augment2(image0)
                image1 = image1.to(self.device)
                image2 = image2.to(self.device)

                if self.writer is not None and niter == 0:
                    grid = torchvision.utils.make_grid(image0[:32])
                    self.writer.add_image('image0', grid, global_step=niter)
                    grid = torchvision.utils.make_grid(image1[:32])
                    self.writer.add_image('image1', grid, global_step=niter)
                    grid = torchvision.utils.make_grid(image2[:32])
                    self.writer.add_image('image2', grid, global_step=niter)

                rep1, _ = self.online_encoder(image1)
                rep2, _ = self.online_encoder(image2)
                online_proj1, online_proj2 = self.online_projector(rep1), self.online_projector(rep2)
                online_pred1 = self.online_predictor(online_proj1)
                online_pred2 = self.online_predictor(online_proj2)

                with torch.no_grad():
                    target_rep1, _ = self.target_encoder(image1)
                    target_rep2, _ = self.target_encoder(image2)
                    target_proj1, target_proj2 = self.target_projector(target_rep1), self.target_projector(target_rep2)
                    target_proj1.detach_()
                    target_proj2.detach_()

                d12 = self.pdict(online_pred1, target_proj2.detach())
                d21 = self.pdict(online_pred2, target_proj1.detach())
#                a_one = self.pairwise_angle(online_pred1, target_proj2.detach())
#                a_two = self.pairwise_angle(online_pred2, target_proj1.detach())
                a12 = pairwise_angle(online_pred1, target_proj2.detach())
                a21 = pairwise_angle(online_pred2, target_proj1.detach())

#                loss_one = loss_fn(online_pred1, target_proj2.detach())
#                loss_two = loss_fn(online_pred2, target_proj1.detach())

#                loss = (loss_one + loss_two).mean()
                d = (d12 + d21).mean()
                a = (a12 + a21).mean()
                loss =  d + a*self.alpha
                d_global = reduce_mean(d, self.distributed) if self.distributed is not None else d.detach()
                a_global = reduce_mean(a, self.distributed) if self.distributed is not None else a.detach()
                self.alpha = d_global.item() / max(a_global.item(), 1e-6)

                self.optimizer.zero_grad()
                loss.backward()
                if self.writer is not None:
                    self.writer.add_scalar('total loss', loss, global_step=niter)
                    self.writer.add_scalar('distance loss', d, global_step=niter)
                    self.writer.add_scalar('angle loss', a, global_step=niter)
                self.optimizer.step()
                loss_epoch += loss.detach()
                lr_epoch += self.optimizer.param_groups[0]['lr']
                update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)
                update_moving_average(self.target_ema_updater, self.target_projector, self.online_projector)
                if self.is_main_process:
                    progress.set_postfix(loss=f'{loss.item():.4f}')
                niter += 1

            loss_epoch /= len(train_loader)
            lr_epoch /= len(train_loader)
            loss_epoch_reduced = reduce_mean(loss_epoch, self.distributed) if self.distributed is not None else loss_epoch.detach()
            self.scheduler.step(loss_epoch_reduced.item())
            if self.writer is not None:
                self.writer.add_scalar('epoch loss', loss_epoch_reduced, global_step=epoch_counter+1)
                self.writer.add_scalar('epoch learing rate', lr_epoch, global_step=epoch_counter+1)
                torch.save({
                    'online_encoder_state_dict': unwrap_module(self.online_encoder).state_dict(),
                    'online_projector_state_dict': unwrap_module(self.online_projector).state_dict(),
                    'online_predictor_state_dict': unwrap_module(self.online_predictor).state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                },(os.path.join(model_checkpoints_folder, 'model_epoch'+str(epoch_counter+1)+'.pth')))
                print("End of epoch {}".format(epoch_counter+1))

class PixclLearner():
    def __init__(
        self,
        online_encoder, 
        target_encoder, 
        online_instance_projector,
        target_instance_projector,
        online_pixel_projector,
        target_pixel_projector,
        optimizer, 
        scheduler, 
        propagate_pixels, 
        online_predictor, 
        device, 
        augment1, 
        augment2,
        **params
    ):
        self.online_encoder = online_encoder
        self.target_encoder = target_encoder
        self.online_instance_projector = online_instance_projector
        self.target_instance_projector = target_instance_projector
        self.online_pixel_projector = online_pixel_projector
        self.target_pixel_projector = target_pixel_projector
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.propagate_pixels = propagate_pixels
        self.online_predictor = online_predictor
        self.device = device
        self.augment1 = augment1
        self.augment2 = augment2
        self.image_sizeH = params['image_sizeH']
        self.image_sizeW = params['image_sizeW']
        self.projection_size = params['projection_size']
        self.projection_hidden_size = params['projection_hidden_size']
        self.target_ema_updater = EMA(params['moving_average_decay'])
        self.distance_thres = params['distance_thres']
        self.similarity_temperature = params['similarity_temperature']
        self.alpha = params['alpha']
        self.alpha_instance=1.
        self.cutout_ratio_range = (0.6, 0.8)
        self.cutout_interpolate_mode = 'nearest' 
        self.coord_cutout_interpolate_mode = 'bilinear'
        self.batch_size = params['batch_size']
        self.max_epochs = params['max_epochs']
        self.pdict = nn.PairwiseDistance(p=2.0)
        self.num_workers = resolve_num_workers(params['num_workers'])
        self.distributed = params.get('distributed')
        self.is_main_process = self.distributed is None or self.distributed.is_main_process
        self.writer = None
        if self.is_main_process:
            self.writer = SummaryWriter(log_dir=os.path.join('runs','pixcl_'+datetime.now().strftime("%Y%m%d-%H%M%S")))
            _create_model_training_folder(self.writer, files_to_same=["./config_pixcl.yaml","./train_pixcl.py","./utils/pixcl_multi.py"])
        # Cache the normalized coordinate grid because image and projection
        # shapes stay constant across batches for a given run.
        self._coordinate_template_cache = {}

    def _get_coordinate_template(self, image_shape, proj_shape, device, dtype):
        cache_key = (image_shape, proj_shape, device.type, str(dtype))
        cached = self._coordinate_template_cache.get(cache_key)
        if cached is not None:
            return cached

        image_h, image_w = image_shape
        proj_image_h, proj_image_w = proj_shape
        coordinates = torch.meshgrid(
            torch.arange(image_h, device=device, dtype=dtype),
            torch.arange(image_w, device=device, dtype=dtype),
            indexing='ij',
        )
        coordinates = torch.stack(coordinates).unsqueeze(0)
        coordinates /= sqrt(image_h ** 2 + image_w ** 2)
        coordinates[:, 0] *= proj_image_h
        coordinates[:, 1] *= proj_image_w
        self._coordinate_template_cache[cache_key] = coordinates
        return coordinates

    def _positive_masks(self, coordinate_template, cutout_coordinates_one, cutout_coordinates_two, proj_shape):
        proj_coors_one = cutout_and_resize(
            coordinate_template,
            cutout_coordinates_one,
            output_size=proj_shape,
            mode=self.coord_cutout_interpolate_mode,
        )
        proj_coors_two = cutout_and_resize(
            coordinate_template,
            cutout_coordinates_two,
            output_size=proj_shape,
            mode=self.coord_cutout_interpolate_mode,
        )

        proj_coors_one = rearrange(proj_coors_one, 'b c h w -> b (h w) c')
        proj_coors_two = rearrange(proj_coors_two, 'b c h w -> b (h w) c')
        # `cdist` avoids explicitly expanding every pixel pair into a flattened
        # `(num_pixels * num_pixels, 2)` tensor before measuring distances.
        distance_matrix = torch.cdist(proj_coors_one, proj_coors_two, p=2).squeeze(0)
        positive_mask_one_two = (distance_matrix < self.distance_thres).to(self.device)
        return positive_mask_one_two, positive_mask_one_two.t()

    def train(self, train_dataset):
        train_sampler = None
        if self.distributed is not None and self.distributed.enabled:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.distributed.world_size,
                rank=self.distributed.rank,
                shuffle=True,
                drop_last=False,
            )

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                num_workers=self.num_workers, drop_last=False, shuffle=train_sampler is None,
                sampler=train_sampler, pin_memory=should_pin_memory(self.device))

        niter = 0
        model_checkpoints_folder = None
        if self.writer is not None:
            model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        for epoch_counter in range(self.max_epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch_counter)
            loss_epoch = 0
            lr_epoch = 0
            progress = tqdm(
                train_loader,
                total=len(train_loader),
                desc=f'PIXCL {epoch_counter + 1}/{self.max_epochs}',
                unit='batch',
                disable=not self.is_main_process,
                leave=False,
            )
            for batch in progress:
                image0 = batch['image']

                # data augmentation to generate two views from the original image
                cutout_coordinates_one, _ = cutout_coordinateW(image0, self.cutout_ratio_range)
                cutout_coordinates_two, _ = cutout_coordinateW(image0, self.cutout_ratio_range)
                #x [B, C, H, W]
                image1_cutout = cutout_and_resize(image0, cutout_coordinates_one, mode = self.cutout_interpolate_mode)
                image2_cutout = cutout_and_resize(image0, cutout_coordinates_two, mode = self.cutout_interpolate_mode)
                # image_xxx_cutout [B, C, H, W]
                image1_cutout, image2_cutout = self.augment1(image1_cutout), self.augment2(image2_cutout)
                image1_cutout = image1_cutout.to(self.device)
                image2_cutout = image2_cutout.to(self.device)

                if self.writer is not None and niter == 0:
                    grid = torchvision.utils.make_grid(image0[:32])
                    self.writer.add_image('image0', grid, global_step=niter)

                    grid = torchvision.utils.make_grid(image1_cutout[:32])
                    self.writer.add_image('image1', grid, global_step=niter)

                    grid = torchvision.utils.make_grid(image2_cutout[:32])
                    self.writer.add_image('image2', grid, global_step=niter)
                """
                    make_grid(tensor[:32]) is to arrange 32 images in a row
                    if batch size is less than 32 output all images in a mini-batch
                    if the batch size is more than 32 output the first 32 images of a mini-batch
                """
                # output two layers (c6 & c5) from the online_encoder, i.e. resnet
                instance1, pixel1 = self.online_encoder(image1_cutout)
                instance2, pixel2 = self.online_encoder(image2_cutout)

                # applied projectors for the two layers
                proj_instance1 = self.online_instance_projector(instance1)
                proj_pixel1 = self.online_pixel_projector(pixel1)
                proj_instance2 = self.online_instance_projector(instance2)
                proj_pixel2 = self.online_pixel_projector(pixel2)

                proj_image_shape = proj_pixel1.shape[2:]
                coordinate_template = self._get_coordinate_template(
                    image_shape=image0.shape[2:],
                    proj_shape=proj_image_shape,
                    device=image0.device,
                    dtype=image0.dtype,
                )
                positive_mask_one_two, positive_mask_two_one = self._positive_masks(
                    coordinate_template,
                    cutout_coordinates_one,
                    cutout_coordinates_two,
                    proj_image_shape,
                )

                # applied target_encoder to output two layers (c6 & c5)
                # obtain target projectors on the two layers
                with torch.no_grad():
                    target_instance1, target_pixel1 = self.target_encoder(image1_cutout)
                    target_instance2, target_pixel2 = self.target_encoder(image2_cutout)
                    target_proj_instance1 = self.target_instance_projector(target_instance1)
                    target_proj_pixel1 = self.target_pixel_projector(target_pixel1)
                    target_proj_instance2 = self.target_instance_projector(target_instance2)
                    target_proj_pixel2 = self.target_pixel_projector(target_pixel2)

                # flatten all the pixel projections
                flatten = lambda t: rearrange(t, 'b c h w -> b c (h w)')
                target_proj_pixel1, target_proj_pixel2 = list(map(flatten, (target_proj_pixel1, target_proj_pixel2)))
                # target_proj_pixel_xxx [B, projection_size, 3x4]

                # applied online_predictor on the instance projection
                # get instance level loss
                pred_instance1 = self.online_predictor(proj_instance1)
                pred_instance2 = self.online_predictor(proj_instance2)
                # pred_instance_xxx [B, projection_size]
#                loss_instance_one_two = loss_fn(pred_instance1, target_proj_instance2.detach())
#                loss_instance_two_one = loss_fn(pred_instance2, target_proj_instance1.detach())
                # loss_instance_xxx [B]
#                instance_loss = (loss_instance_one_two + loss_instance_two_one).mean()

                d12 = self.pdict(pred_instance1, target_proj_instance2.detach())
                d21 = self.pdict(pred_instance2, target_proj_instance1.detach())
                a12 = pairwise_angle(pred_instance1, target_proj_instance2.detach())
                a21 = pairwise_angle(pred_instance2, target_proj_instance1.detach())
                d = (d12 + d21).mean()
                a = (a12 + a21).mean()
                instance_loss =  d + a*self.alpha_instance
                d_global = reduce_mean(d, self.distributed) if self.distributed is not None else d.detach()
                a_global = reduce_mean(a, self.distributed) if self.distributed is not None else a.detach()
                self.alpha_instance = d_global.item() / max(a_global.item(), 1e-6)

                # applied pixel propagator on the pixel projection 
                # calculate pix pro loss
                propagated_pixels1 = self.propagate_pixels(proj_pixel1)
                propagated_pixels2 = self.propagate_pixels(proj_pixel2)
                # propagated_pixels_xxx [B, projection_size, 3, 4]
                propagated_pixels1, propagated_pixels2 = list(map(flatten, (propagated_pixels1, propagated_pixels2)))
                # propagated_pixels_xxx [B, projection_size, 12]

                propagated_similarity_one_two = F.cosine_similarity(propagated_pixels1[..., :, None], target_proj_pixel2[..., None, :], dim = 1)
                propagated_similarity_two_one = F.cosine_similarity(propagated_pixels2[..., :, None], target_proj_pixel1[..., None, :], dim = 1)
                #propagated_similarity_xxx_xxx [B, 12, 12]

                loss_pixpro_one_two = propagated_similarity_one_two.masked_select(positive_mask_one_two[None, ...]).mean()
                loss_pixpro_two_one = propagated_similarity_two_one.masked_select(positive_mask_two_one[None, ...]).mean()
                pixpro_loss = 2 - loss_pixpro_one_two - loss_pixpro_two_one

                # total loss
                loss = pixpro_loss*self.alpha + instance_loss
                pixpro_global = reduce_mean(pixpro_loss, self.distributed) if self.distributed is not None else pixpro_loss.detach()
                instance_loss_global = reduce_mean(instance_loss, self.distributed) if self.distributed is not None else instance_loss.detach()
                self.alpha = instance_loss_global.item() / max(pixpro_global.item(), 1e-6)
        
                self.optimizer.zero_grad()
                loss.backward()
                if self.writer is not None:
                    self.writer.add_scalar('loss_total', loss, global_step=niter)
                    self.writer.add_scalar('loss_pixpro', pixpro_loss, global_step=niter)
                    self.writer.add_scalar('loss_instance', instance_loss, global_step=niter)
                    self.writer.add_scalar('loss_dist', d, global_step=niter)
                    self.writer.add_scalar('loss_angle', a, global_step=niter)
                self.optimizer.step()
                update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)
                update_moving_average(self.target_ema_updater, self.target_instance_projector, self.online_instance_projector)
                update_moving_average(self.target_ema_updater, self.target_pixel_projector, self.online_pixel_projector)
                loss_epoch += loss.detach()
                lr_epoch += self.optimizer.param_groups[0]['lr']
                if self.is_main_process:
                    progress.set_postfix(loss=f'{loss.item():.4f}')
                niter += 1

            loss_epoch /= len(train_loader)
            lr_epoch /= len(train_loader)
            loss_epoch_reduced = reduce_mean(loss_epoch, self.distributed) if self.distributed is not None else loss_epoch.detach()
            self.scheduler.step(loss_epoch_reduced.item())
            if self.writer is not None:
                self.writer.add_scalar('epoch loss', loss_epoch_reduced, global_step=epoch_counter+1)
                self.writer.add_scalar('epoch learning rate', lr_epoch, global_step=epoch_counter+1)
                torch.save({
                    'online_encoder_state_dict': unwrap_module(self.online_encoder).state_dict(),
                    'online_instance_projector_state_dict': unwrap_module(self.online_instance_projector).state_dict(),
                    'online_pixel_projector_state_dict': unwrap_module(self.online_pixel_projector).state_dict(),
                    'propagate_pixels_state_dict': unwrap_module(self.propagate_pixels).state_dict(),
                    'online_predictor_state_dict': unwrap_module(self.online_predictor).state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                },(os.path.join(model_checkpoints_folder, 'model_epoch'+str(epoch_counter+1)+'.pth')))
                print("End of epoch {}".format(epoch_counter+1))
