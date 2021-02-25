import sys
import yaml
from argparse import ArgumentParser
from fommia.data.dataset import FramesDataset
from fommia.modules.generator import OcclusionAwareGenerator
from fommia.modules.keypoint_detector import KPDetector
from fommia.animate import Animator

if __name__ == "__main__":

    if sys.version_info[0] < 3:
        raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

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

    # Animate
    Animator(config=config,
             generator=OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                               **config['model_params']['common_params']),
             kp_detector=KPDetector(**config['model_params']['kp_detector_params'],
                                    **config['model_params']['common_params']),
             checkpoint=opt.checkpoint,
             dataset=FramesDataset(is_train=(opt.mode == 'train'), **config['dataset_params']),
             device_ids=opt.device_ids,
             verbose=opt.verbose)
    Animator.run()
