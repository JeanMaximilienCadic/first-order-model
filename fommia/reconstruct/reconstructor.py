import os
import torch
from tqdm import tqdm
import numpy as np
import imageio
from time import gmtime, strftime
from shutil import copy
from torch.utils.data import DataLoader
from fommia.trainers.logger import Visualizer


class Reconstructor:
    def __init__(self, config, generator, discriminator, kp_detector, checkpoint, dataset, device_ids, verbose=False):
        self._config = config
        self._generator = generator
        self._discriminator = discriminator
        self._kp_detector = kp_detector
        self._checkpoint = checkpoint
        self._dataset = dataset
        self._device_ids = device_ids
        self._verbose = verbose
        self._loss_list = []
        self.init()

    def init(self):
        if self._checkpoint is not None:
            self._log_dir = os.path.join(*os.path.split(self._checkpoint)[:-1])
        else:
            self._log_dir = os.path.join(self._log_dir, os.path.basename(self._config).split('.')[0])
            self._log_dir += ' ' + strftime("%d_%m_%y_%H.%M.%S", gmtime())

        if torch.cuda.is_available():
            self._generator.to(self._device_ids[0])
            self._discriminator.to(self._device_ids[0])
            self._kp_detector.to(self._device_ids[0])

        if not os.path.exists(self._log_dir):
            os.makedirs(self._log_dir)
        if not os.path.exists(os.path.join(self._log_dir, os.path.basename(self._opt.config))):
            copy(self._config, self._log_dir)

        self._png_dir = os.path.join(self._log_dir, 'reconstruction/png')
        self._log_dir = os.path.join(self._log_dir, 'reconstruction')
        self._dataloader = DataLoader(self._dataset, batch_size=1, shuffle=False, num_workers=1)
        self._generator.eval()
        self._kp_detector.eval()

        if self._verbose:
            print(self._kp_detector)
            print(self._generator)
            print(self._discriminator)

    def run(self):
        for it, x in tqdm(enumerate(self._dataloader)):
            if self._config['reconstruction_params']['num_videos'] is not None:
                if it > self._config['reconstruction_params']['num_videos']:
                    break
            with torch.no_grad():
                predictions = []
                visualizations = []
                if torch.cuda.is_available():
                    x['video'] = x['video'].cuda()
                kp_source = self._kp_detector(x['video'][:, :, 0])
                for frame_idx in range(x['video'].shape[2]):
                    source = x['video'][:, :, 0]
                    driving = x['video'][:, :, frame_idx]
                    kp_driving = self._kp_detector(driving)
                    out = self._generator(source, kp_source=kp_source, kp_driving=kp_driving)
                    out['kp_source'] = kp_source
                    out['kp_driving'] = kp_driving
                    del out['sparse_deformed']
                    predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])

                    visualization = Visualizer(**self._config['visualizer_params']).visualize(source=source,
                                                                                        driving=driving, out=out)
                    visualizations.append(visualization)

                    self._loss_list.append(torch.abs(out['prediction'] - driving).mean().cpu().numpy())

                predictions = np.concatenate(predictions, axis=1)
                imageio.imsave(os.path.join(self._png_dir, x['name'][0] + '.png'), (255 * predictions).astype(np.uint8))

                image_name = x['name'][0] + self._config['reconstruction_params']['format']
                imageio.mimsave(os.path.join(self._log_dir, image_name), visualizations)

        print("Reconstruction loss: %s" % np.mean(self._loss_list))
