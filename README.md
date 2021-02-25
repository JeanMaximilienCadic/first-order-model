# First Order Motion Model for Image Animation

This repository contains the source code for the paper [First Order Motion Model for Image Animation](https://papers.nips.cc/paper/8935-first-order-motion-model-for-image-animation) by Aliaksandr Siarohin, [Stéphane Lathuilière](http://stelat.eu), [Sergey Tulyakov](http://stulyakov.com), [Elisa Ricci](http://elisaricci.eu/) and [Nicu Sebe](http://disi.unitn.it/~sebe/). 

![Screenshot](imgs/dicaprio.gif)


### Datasets

1) **Bair**. This dataset can be directly [downloaded](https://yadi.sk/d/Rr-fjn-PdmmqeA).

2) **Mgif**. This dataset can be directly [downloaded](https://yadi.sk/d/5VdqLARizmnj3Q).

3) **Fashion**. Follow the instruction on dataset downloading [from](https://vision.cs.ubc.ca/datasets/fashion/).

4) **Taichi**. Follow the instructions in [data/taichi-loading](fommia/data/taichi-loading/README.md) or instructions from https://github.com/AliaksandrSiarohin/video-preprocessing. 

5) **Nemo**. Please follow the [instructions](https://www.uva-nemo.org/) on how to download the dataset. Then the dataset should be preprocessed using scripts from https://github.com/AliaksandrSiarohin/video-preprocessing.
 
6) **VoxCeleb**. Please follow the instruction from https://github.com/AliaksandrSiarohin/video-preprocessing.

7) **Custom** 

    1) Resize all the videos to the same size e.g 256x256, the videos can be in '.gif', '.mp4' or folder with images.
We recommend the later, for each video make a separate folder with all the frames in '.png' format. This format is loss-less, and it has better i/o performance.

    2) Create a folder ```data/dataset_name``` with 2 subfolders ```train``` and ```test```, put training videos in the ```train``` and testing in the ```test```.

    3) Create a config ```config/dataset_name.yaml```, in dataset_params specify the root dir the ```root_dir:  data/dataset_name```. Also adjust the number of epoch in train_params.

### Pre-trained checkpoint
Checkpoints can be found under following link: [google-drive](https://drive.google.com/open?id=1PyQJmkdCsAkOYwUyaj_l-l0as-iLDgeH) or [yandex-disk](https://yadi.sk/d/lEw8uRm140L_eQ).

### Installation

We support ```python3```. To install the dependencies run:
```
pip install -r requirements.txt
```

### YAML configs

There are several configuration (```config/dataset_name.yaml```) files one for each `dataset`. See ```config/taichi-256.yaml``` to get description of each parameter.

### Test the setup
```
python -m fommia.test
```

```
=1= TEST PASSED : fommia
=1= TEST PASSED : fommia.animate
=1= TEST PASSED : fommia.data
=1= TEST PASSED : fommia.data.dataset
=1= TEST PASSED : fommia.modules
=1= TEST PASSED : fommia.reconstruct
=1= TEST PASSED : fommia.test
=1= TEST PASSED : fommia.trainers
```

### Animation Demo
To run a demo, download checkpoint and run the following command:
```
python -W ignore -m fommia \
  --config vox-256 \
  --driving_video __data__/08.mp4 \
  --source_image __data__/01.png  \
  --checkpoint __data__/pth/vox-cpk.pth.tar \
  --result_video result.mp4  \
  --relative \
  --adapt_scale
```
The result will be stored in ```result.mp4```.

### Training

To train a model on specific dataset run:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m fommia.trainers \
	--config vox-256 \
	--device_ids 0,1,2,3
```

This will execute the following script:
```
import yaml
from argparse import ArgumentParser
from fommia.data.dataset import FramesDataset
from fommia.modules.generator import OcclusionAwareGenerator
from fommia.modules.discriminator import MultiScaleDiscriminator
from fommia.modules.keypoint_detector import KPDetector
from fommia.trainers import FOMMIATrainer

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--mode", default="train", choices=["train", "reconstruction", "animate"])
    parser.add_argument("--log_dir", default='log', help="path to log into")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")
    parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print model architecture")
    parser.set_defaults(verbose=False)

    opt = parser.parse_args()
    config = yaml.load(open(opt.config))
    model_params, dataset_params =config['dataset_params'], config['model_params']

    # Trainer
    trainer = FOMMIATrainer(config=config,
                            generator=OcclusionAwareGenerator(**model_params['generator_params'],
                                                              **model_params['common_params']),
                            discriminator=MultiScaleDiscriminator(**model_params['discriminator_params'],
                                                                  **model_params['common_params']),
                            kp_detector=KPDetector(**model_params['kp_detector_params'],
                                                   **model_params['common_params']),
                            checkpoint=opt.checkpoint,
                            dataset=FramesDataset(is_train=True, **dataset_params),
                            device_ids=opt.device_ids)
    trainer.run()
```
The code will create a folder in the log directory (each run will create a time-stamped new directory).
Checkpoints will be saved to this folder.
To check the loss values during training see ```log.txt```.
You can also check training data reconstructions in the ```train-vis``` subfolder.
By default the batch size is tunned to run on 2 or 4 Titan-X gpu (appart from speed it does not make much difference). You can change the batch size in the train_params in corresponding ```.yaml``` file.

### Evaluation on video reconstruction

To evaluate the reconstruction performance run:
```
CUDA_VISIBLE_DEVICES=0 \
python -m fommia.reconstruct \
        --config dataset_name \
        --checkpoint fommia.ckpt
```
You will need to specify the path to the checkpoint,
the ```reconstruction``` subfolder will be created in the checkpoint folder.
The generated video will be stored to this folder, also generated videos will be stored in ```png``` subfolder in loss-less '.png' format for evaluation.
Instructions for computing metrics from the paper can be found: https://github.com/AliaksandrSiarohin/pose-evaluation.

This will execute the following script
```
import yaml
from argparse import ArgumentParser
from fommia.data.dataset import FramesDataset

from fommia.modules.generator import OcclusionAwareGenerator
from fommia.modules.discriminator import MultiScaleDiscriminator
from fommia.modules.keypoint_detector import KPDetector
from fommia.reconstruct import Reconstructor

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--mode", default="train", choices=["train", "reconstruction", "animate"])
    parser.add_argument("--log_dir", default='log', help="path to log into")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")
    parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print model architecture")
    parser.set_defaults(verbose=False)

    opt = parser.parse_args()
    config = yaml.load(open(opt.config))
    model_params, dataset_params =config['dataset_params'], config['model_params']

    # Reconstructor
    reconstructor = Reconstructor(config,
                                  generator = OcclusionAwareGenerator(**model_params['generator_params'],
                                                                      **model_params['common_params']),
                                  discriminator = MultiScaleDiscriminator(**model_params['discriminator_params'],
                                                                          **model_params['common_params']),
                                  kp_detector = KPDetector(**model_params['kp_detector_params'],
                                                           **model_params['common_params']),
                                  checkpoint=opt.checkpoint,
                                  dataset=FramesDataset(**dataset_params))

```
### Image animation

In order to animate videos run:
```
CUDA_VISIBLE_DEVICES=0 \
python -m fommia.animate \
        --config dataset_name \
        --checkpoint fommia.ckpt
```
You will need to specify the path to the checkpoint,
the ```animation``` subfolder will be created in the same folder as the checkpoint.
You can find the generated video there and its loss-less version in the ```png``` subfolder.
By default video from test set will be randomly paired, but you can specify the "source,driving" pairs in the corresponding ```.csv``` files. The path to this file should be specified in corresponding ```.yaml``` file in pairs_list setting.

There are 2 different ways of performing animation:
by using **absolute** keypoint locations or by using **relative** keypoint locations.

1) <i>Animation using absolute coordinates:</i> the animation is performed using the absolute postions of the driving video and appearance of the source image.
In this way there are no specific requirements for the driving video and source appearance that is used.
However this usually leads to poor performance since unrelevant details such as shape is transfered.
Check animate parameters in ```taichi-256.yaml``` to enable this mode.

<img src="imgs/absolute-demo.gif" width="512"> 

2) <i>Animation using relative coordinates:</i> from the driving video we first estimate the relative movement of each keypoint,
then we add this movement to the absolute position of keypoints in the source image.
This keypoint along with source image is used for animation. This usually leads to better performance, however this requires
that the object in the first frame of the video and in the source image have the same pose

<img src="imgs/relative-demo.gif" width="512"> 

This will execute the following script:
```
import sys
import yaml
from argparse import ArgumentParser
from fommia.data.dataset import FramesDataset
from fommia.modules.generator import OcclusionAwareGenerator
from fommia.modules.keypoint_detector import KPDetector
from fommia.animate import Animator

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--log_dir", default='log', help="path to log into")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")
    parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print model architecture")
    parser.set_defaults(verbose=False)

    opt = parser.parse_args()
    config = yaml.load(open(opt.config))
    model_params, dataset_params =config['dataset_params'], config['model_params']

    # Animate
    animator = Animator(config=config,
                        generator=OcclusionAwareGenerator(**model_params['generator_params'],
                                                          **model_params['common_params']),
                        kp_detector=KPDetector(**model_params['kp_detector_params'],
                                               **model_params['common_params']),
                        checkpoint=opt.checkpoint,
                        dataset=FramesDataset(**dataset_params),
                        device_ids=opt.device_ids,
                        verbose=opt.verbose)
    animator.run()
```

#### Additional notes

Citation:

```
@InProceedings{Siarohin_2019_NeurIPS,
  author={Siarohin, Aliaksandr and Lathuilière, Stéphane and Tulyakov, Sergey and Ricci, Elisa and Sebe, Nicu},
  title={First Order Motion Model for Image Animation},
  booktitle = {Conference on Neural Information Processing Systems (NeurIPS)},
  month = {December},
  year = {2019}
}
```

git clone https://github.com/facebookresearch/maskrcnn-benchmark