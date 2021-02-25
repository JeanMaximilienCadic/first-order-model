import os
import torch
from tqdm import tqdm
import numpy as np
import imageio
from time import gmtime, strftime
from shutil import copy
from fommia.animate import normalize_kp
from torch.utils.data import DataLoader
from fommia.data.dataset import PairedDataset
from fommia.trainers.logger import Logger, Visualizer
from fommia.data import DataParallelWithCallback

class Animator:
    def __init__(self, config, generator, kp_detector, checkpoint, dataset, device_ids, verbose=False):
        self._config = config
        self._generator = generator
        self._kp_detector = kp_detector
        self._checkpoint = checkpoint
        self._dataset = dataset
        self._device_ids = device_ids
        self._verbose = verbose
        self.init()

    def init(self):
        if self._checkpoint is not None:
            self._log_dir = os.path.join(*os.path.split(self._checkpoint)[:-1])
        else:
            self._log_dir = os.path.join(self._log_dir, os.path.basename(self._config).split('.')[0])
            self._log_dir += ' ' + strftime("%d_%m_%y_%H.%M.%S", gmtime())

        if not os.path.exists(self._log_dir):
            os.makedirs(self._log_dir)
        if not os.path.exists(os.path.join(self._log_dir, os.path.basename(self._config))):
            copy(self._config, self._log_dir)

        if torch.cuda.is_available():
            self._kp_detector.to(self._device_ids[0])
            self._generator.to(self._device_ids[0])
        if self._verbose:
            print(self._generator)
            print(self._kp_detector)



    def run(self):
        log_dir = os.path.join(self._log_dir, 'animation')
        png_dir = os.path.join(self._log_dir, 'png')
        animate_params = self._config['animate_params']

        dataset = PairedDataset(initial_dataset=self._dataset, number_of_pairs=animate_params['num_pairs'])
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

        if self._checkpoint is not None:
            Logger.load_cpk(self._checkpoint, generator=self._generator, kp_detector=self._kp_detector)
        else:
            raise AttributeError("Checkpoint should be specified for mode='animate'.")

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        if not os.path.exists(png_dir):
            os.makedirs(png_dir)

        if torch.cuda.is_available():
            generator = DataParallelWithCallback(self._generator)
            kp_detector = DataParallelWithCallback(self._kp_detector)

        generator.eval()
        kp_detector.eval()

        for it, x in tqdm(enumerate(dataloader)):
            with torch.no_grad():
                predictions = []
                visualizations = []

                driving_video = x['driving_video']
                source_frame = x['source_video'][:, :, 0, :, :]

                kp_source = kp_detector(source_frame)
                kp_driving_initial = kp_detector(driving_video[:, :, 0])

                for frame_idx in range(driving_video.shape[2]):
                    driving_frame = driving_video[:, :, frame_idx]
                    kp_driving = kp_detector(driving_frame)
                    kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                           kp_driving_initial=kp_driving_initial, **animate_params['normalization_params'])
                    out = generator(source_frame, kp_source=kp_source, kp_driving=kp_norm)

                    out['kp_driving'] = kp_driving
                    out['kp_source'] = kp_source
                    out['kp_norm'] = kp_norm

                    del out['sparse_deformed']

                    predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])

                    visualization = Visualizer(**self._config['visualizer_params']).visualize(source=source_frame,
                                                                                        driving=driving_frame, out=out)
                    visualization = visualization
                    visualizations.append(visualization)

                predictions = np.concatenate(predictions, axis=1)
                result_name = "-".join([x['driving_name'][0], x['source_name'][0]])
                imageio.imsave(os.path.join(png_dir, result_name + '.png'), (255 * predictions).astype(np.uint8))

                image_name = result_name + animate_params['format']
                imageio.mimsave(os.path.join(log_dir, image_name), visualizations)
