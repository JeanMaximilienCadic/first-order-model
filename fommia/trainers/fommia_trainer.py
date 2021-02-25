from tqdm import trange
from fommia.modules.model import GeneratorFullModel, DiscriminatorFullModel
from torch.optim.lr_scheduler import MultiStepLR
from fommia.data.dataset import DatasetRepeater
import os
import torch
from time import gmtime, strftime
from shutil import copy
from torch.utils.data import DataLoader
from fommia.data.dataset import PairedDataset
from fommia.trainers import Logger
from fommia.data import DataParallelWithCallback


class FOMMIATrainer:
    def __init__(self, config, generator, discriminator, kp_detector, checkpoint, dataset, device_ids, verbose):
        self._config = config
        self._generator = generator
        self._discriminator = discriminator
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

        self._log_dir = os.path.join(self._log_dir, 'animation')
        self._png_dir = os.path.join(self._log_dir, 'png')
        [os.makedirs(d) for d in [self._log_dir, self._png_dir] if not os.path.exists(d)]

        if not os.path.exists(os.path.join(self._log_dir, os.path.basename(self._config))):
            copy(self._config, self._log_dir)

        self._animate_params = self._config['animate_params']
        self._dataset = PairedDataset(initial_dataset=self._dataset, number_of_pairs=self._animate_params['num_pairs'])
        self._dataloader = DataLoader(self._dataset, batch_size=1, shuffle=False, num_workers=1)

        if self._checkpoint is not None:
            Logger.load_cpk(self._checkpoint, generator=self._generator, kp_detector=self._kp_detector)
        else:
            raise AttributeError("Checkpoint should be specified for mode='animate'.")
        self._generator.eval()
        self._kp_detector.eval()

        self._train_params = self._config['train_params']
        self._optimizer_generator = torch.optim.Adam(self._generator.parameters(),
                                                     lr=self._train_params['lr_generator'],
                                                     betas=(0.5, 0.999))
        self._optimizer_discriminator = torch.optim.Adam(self._discriminator.parameters(),
                                                         lr=self._train_params['lr_discriminator'],
                                                         betas=(0.5, 0.999))
        self._optimizer_kp_detector = torch.optim.Adam(self._kp_detector.parameters(),
                                                       lr=self._train_params['lr_kp_detector'],
                                                       betas=(0.5, 0.999))

        if self._checkpoint is not None:
            self._start_epoch = Logger.load_cpk(self._checkpoint, self._generator, self._discriminator, self._kp_detector,
                                          self._optimizer_generator, self._optimizer_discriminator,
                                          None if self._train_params['lr_kp_detector'] == 0 else self._optimizer_kp_detector)
        else:
            self._start_epoch = 0

        self._scheduler_generator = MultiStepLR(self._optimizer_generator,
                                                self._train_params['epoch_milestones'],
                                                gamma=0.1,
                                                last_epoch=self._start_epoch - 1)
        self._scheduler_discriminator = MultiStepLR(self._optimizer_discriminator,
                                                    self._train_params['epoch_milestones'],
                                                    gamma=0.1,
                                                    last_epoch=self._start_epoch - 1)
        self._scheduler_kp_detector = MultiStepLR(self._optimizer_kp_detector,
                                                  self._train_params['epoch_milestones'],
                                                  gamma=0.1,
                                                  last_epoch=-1 + self._start_epoch * (self._train_params['lr_kp_detector'] != 0))

        if 'num_repeats' in self._train_params or self._train_params['num_repeats'] != 1:
            self._dataset = DatasetRepeater(self._dataset, self._train_params['num_repeats'])
        self._dataloader = DataLoader(self._dataset, batch_size=self._train_params['batch_size'], shuffle=True, num_workers=6, drop_last=True)

        self._generator_full = GeneratorFullModel(self._kp_detector, self._generator, self._discriminator, self._train_params)
        self._discriminator_full = DiscriminatorFullModel(self._kp_detector, self._generator, self._discriminator, self._train_params)

        if torch.cuda.is_available():
            self._kp_detector.to(self._device_ids[0])
            self._generator.to(self._device_ids[0])
            self._generator = DataParallelWithCallback(self._generator)
            self._kp_detector = DataParallelWithCallback(self._kp_detector)
            self._generator_full = DataParallelWithCallback(self._generator_full, device_ids=self._device_ids)
            self._discriminator_full = DataParallelWithCallback(self._discriminator_full, device_ids=self._device_ids)
        if self._verbose:
            print(self._generator)
            print(self._kp_detector)


    def run(self):
        with Logger(log_dir=self._log_dir, visualizer_params=self._config['visualizer_params'], checkpoint_freq=self._train_params['checkpoint_freq']) as logger:
            for epoch in trange(self._start_epoch, self._train_params['num_epochs']):
                for x in self._dataloader:
                    losses_generator, generated = self._generator_full(x)

                    loss_values = [val.mean() for val in losses_generator.values()]
                    loss = sum(loss_values)

                    loss.backward()
                    self._optimizer_generator.step()
                    self._optimizer_generator.zero_grad()
                    self._optimizer_kp_detector.step()
                    self._optimizer_kp_detector.zero_grad()

                    if self._train_params['loss_weights']['generator_gan'] != 0:
                        self._optimizer_discriminator.zero_grad()
                        losses_discriminator = self._discriminator_full(x, generated)
                        loss_values = [val.mean() for val in losses_discriminator.values()]
                        loss = sum(loss_values)

                        loss.backward()
                        self._optimizer_discriminator.step()
                        self._optimizer_discriminator.zero_grad()
                    else:
                        losses_discriminator = {}

                    losses_generator.update(losses_discriminator)
                    losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                    logger.log_iter(losses=losses)

                self._scheduler_generator.step()
                self._scheduler_discriminator.step()
                self._scheduler_kp_detector.step()

                logger.log_epoch(epoch, {'generator': self._generator,
                                         'discriminator': self._discriminator,
                                         'kp_detector': self._kp_detector,
                                         'optimizer_generator': self._optimizer_generator,
                                         'optimizer_discriminator': self._optimizer_discriminator,
                                         'optimizer_kp_detector': self._optimizer_kp_detector}, inp=x, out=generated)
