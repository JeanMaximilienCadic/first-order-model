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
