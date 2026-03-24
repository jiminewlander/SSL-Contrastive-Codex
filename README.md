# SSL-contrastive
Self-supervised contrastive learning to decipher plasma frequency from space-based wave receivers

Codes have been approved for public release; distribution is unlimited. Public Affairs release approval #AFRL-2024-4503

A manuscript titled "Decoding space radio waves: Self-supervised AI decipher plasma frequency" by Su and Carilli details the construction of the software. 

Please refer to https://doi.org/10.1029/2023RS007907 for the loss functions (binary focal cross-entroy loss, focal Tversky loss, & Hausdorff morphological erosion loss) used in the downstream task and the prediction metrics (Huasdoff Distance)

Self-supervised pretext tasks:
  * BYOL - main program train_byol.py with input parameters specified in config_byol.yaml
  * PIXCL - main program train_picxl.py with input parameters specified in config_pixcl.yaml

Downstream task:
  * FCN - main program downstream_fcn.py with input parameters specified in config_fcn.yaml

Prediction: main program predict_fcn.py with input parameters specified in config_predict.yaml

Other python source codes are located in the "utils" folder
  * data_loader.py, custom_transform.py, pixcl_multi.py are called by train_byol.py and train_pixcl.py
  * data_loader_downstream.py, pixcl_multi.py, fcn.py, hausdorff.py and dice_score.py are called by downstream_fcn.py
  * data_loader_downstream.py, pixcl_multi.py, fcn.py, and hausdorff.py are called by predict_fcn.py

Images used to train, validate, and test for this study is published at https://doi.org/10.5281/zenodo.13351491

The software utilized in this project was obtained, modified, and consolidated from the following sources. Our version was meticulously undertaken to ensure precise alignment with the specific requirements and objectives of our investigation.

BYOL - Bootstrap Your Own Latent 
       
       original architecture described in https://arxiv.org/abs/2006.07733
       sample software https://github.com/lucidrains/byol-pytorch

PIXCL - PIXel-level Consistency Learning 
        
        original architecture described in https://arxiv.org/abs/2011.10043
        sample software https://github.com/lucidrains/pixel-level-contrastive-learning

FCN  - Fully Convolutional Networks
       
       original architecture described in https://arxiv.org/abs/1411.4038
       easiest FCN https://github.com/pochih/FCN-pytorch

## Repository flow

The codebase has a two-stage workflow:

1. Self-supervised pretraining on unlabeled spectrogram images
   * [train_byol.py](./train_byol.py) trains a ResNet backbone with a BYOL-style objective.
   * [train_pixcl.py](./train_pixcl.py) trains the same backbone with a pixel-level consistency objective.
   * Shared SSL components live in [utils/pixcl_multi.py](./utils/pixcl_multi.py).

2. Supervised downstream segmentation
   * [downstream_fcn.py](./downstream_fcn.py) attaches an FCN decoder to the pretrained backbone and trains on image/mask pairs.
   * [predict_fcn.py](./predict_fcn.py) runs inference and writes Hausdorff-based evaluation metrics.
   * FCN decoders live in [utils/fcn.py](./utils/fcn.py).

Supporting modules:

* [utils/data_loader.py](./utils/data_loader.py): image-only dataset for SSL.
* [utils/data_loader_downstream.py](./utils/data_loader_downstream.py): image+mask dataset for segmentation.
* [utils/custom_transform.py](./utils/custom_transform.py): custom augmentations such as jitter, FFT perturbation, and axis warping.
* [utils/dice_score.py](./utils/dice_score.py): Tversky and focal Tversky loss.
* [utils/hausdorff.py](./utils/hausdorff.py): Hausdorff-based loss and metrics.

## Local setup

The repository ships with a dependency manifest in `requirements.txt`, but the config paths are still local-directory based. Before running anything locally:

1. Create an environment and install dependencies.
2. Put your data into the local project directories described below.
3. Run the relevant entry point with the matching config file in the repository root.

Example setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The configs are now set up for this local directory layout:

```text
data/
  pretrain/
    imgs/
  downstream/
    imgs/
    masks/
  predict/
    imgs/
    masks/
outputs/
  predict/
```

Mask files for downstream training and prediction should match the image stem plus the configured suffix `_mask`.
Example: `sample.png` pairs with `sample_mask.png`.

After you finish pretraining or FCN training, update the relevant `pretrain_dir` and `pretrain_epoch` entries in the YAML files to point at the run you want to reuse.

Typical command sequence:

```bash
python train_byol.py
python train_pixcl.py
python downstream_fcn.py
python predict_fcn.py
```

TensorBoard logs and checkpoints are written under `runs/`.

## Apple Silicon notes

The training and prediction scripts now choose the best available backend automatically:

* `cuda` on NVIDIA GPUs
* `mps` on Apple Silicon
* `cpu` otherwise

On Apple Silicon, the code now defaults to `num_workers: 0` unless you override it in the YAML. This is the safest default for PIL-based image loading on macOS. If you want to tune for your machine, start with `0`, then try `2`, and compare epoch time and stability.

Prediction now requires a trained FCN checkpoint. If `pretrain_dir` or `pretrain_epoch` is missing or incorrect in `config_predict.yaml`, the script will stop with an error instead of silently running with random weights.
