"""
  developed by Yi-Jiun Su
  The latest update in April 2024
"""
import yaml
import torch
import torchvision.transforms as T
import utils.custom_transform as CT
from utils.data_loader import CustomDataset
import torchvision
from utils.pixcl_multi import NetWrapperMultiLayers, MLP, BYOLTrainer
from utils.runtime import get_best_device, configure_torch_runtime
import os
from copy import deepcopy

print(torch.__version__)
# controlling sources of randomness
torch.manual_seed(0)

def normalize_optional(value):
    if value in (None, '', 'None', 'none', 'null', 'Null'):
        return None
    return value

if __name__ == '__main__':
    config = yaml.safe_load(open("config_byol.yaml", "r"))
    device = get_best_device()
    configure_torch_runtime(device)
    print(f"Learning with: {device}")
    pretrain_dir = normalize_optional(config.get('pretrain_dir'))
    pretrain_epoch = normalize_optional(config.get('pretrain_epoch'))

    # define data Transform functions
    # T1 = T.Compose([CT.Jitter(),CT.Scale()])
    T1 = T.Compose([CT.Jitter(),CT.FFT()])
    T2 = T.Compose([CT.WrapHW(dim=2),CT.WrapHW(dim=3)])
    dataset = CustomDataset(**config['dataset'])

    if config['backbone_arch'] == 'resnet18':
        resnet = torchvision.models.resnet18()
    elif config['backbone_arch'] == 'resnet34':
        resnet = torchvision.models.resnet34()
    elif config['backbone_arch'] == 'resnet50':
        resnet = torchvision.models.resnet50()

    #online_encoder include instance- (MLP) & pixel-level (ConvMLP) projectors
    online_encoder = NetWrapperMultiLayers(net=resnet).to(device)
# Not working in a virtural environment yet
#    if torch.cuda.device_count() > 1:
#        print("Let's use", torch.cuda.device_count(), "GPUs!")
#        online_encoder = torch.nn.DataParallel(online_encoder)

    with torch.no_grad():
        sample = torch.randn(
            1,
            config['dataset']['channels'],
            config['learner']['image_sizeH'],
            config['learner']['image_sizeW'],
            device=device
        )
        instance_representation, _ = online_encoder(sample)

    online_projector = MLP(
        instance_representation.shape[1],
        config['learner']['projection_size'],
        config['learner']['projection_hidden_size']).to(device)

    # instance level predictor
    online_predictor = MLP(
        config['learner']['projection_size'],
        config['learner']['projection_size'],
        config['learner']['projection_hidden_size']).to(device)
    
    if config['opt_method'] == 'adam_hd':
        raise ValueError(
            "opt_method='adam_hd' is not supported in this repository because "
            "the Adam_HD implementation is not bundled. Use 'adam' or 'radam'."
        )
    elif config['opt_method'] == 'radam':
        opt = torch.optim.RAdam(list(online_encoder.parameters()) +
              list(online_projector.parameters()) +
              list(online_predictor.parameters()),
              **config['optimizer'])
    else: 
        opt = torch.optim.Adam(list(online_encoder.parameters()) +
              list(online_projector.parameters()) +
              list(online_predictor.parameters()),
              **config['optimizer'])

    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,**config['lr_sch_param'])

    if pretrain_dir:
        try:
            checkpoints_dir = os.path.join('./runs', pretrain_dir, 'checkpoints')
            # load pre-trained parameters
            load_params = torch.load(os.path.join(os.path.join(
                checkpoints_dir, 'model_epoch' + pretrain_epoch + '.pth')),
                map_location=torch.device(device))
            online_encoder.load_state_dict(load_params['online_encoder_state_dict'])
            if 'online_projector_state_dict' in load_params:
                online_projector.load_state_dict(load_params['online_projector_state_dict'])
            if 'online_predictor_state_dict' in load_params:
                online_predictor.load_state_dict(load_params['online_predictor_state_dict'])
            try:
                opt.load_state_dict(load_params['optimizer_state_dict'])
            except ValueError:
                print("Optimizer state incompatible with current model. Continuing with a fresh optimizer.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

    with torch.no_grad():
        target_encoder = deepcopy(online_encoder)
        for p in target_encoder.parameters():
            p.requires_grad = False
        target_projector = deepcopy(online_projector)
        for p in target_projector.parameters():
            p.requires_grad = False

    learner = BYOLTrainer(
        online_encoder=online_encoder, 
        target_encoder=target_encoder, 
        online_projector=online_projector,
        target_projector=target_projector,
        online_predictor=online_predictor, 
        optimizer=opt, 
        scheduler=sch, 
        device=device,
        augment1=T1, 
        augment2=T2,
        **config['learner'])

    learner.train(dataset)
