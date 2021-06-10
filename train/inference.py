import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
print(ROOT)

sys.path.append(ROOT)

from external_lib.MedCommon.gan.runner.train_gan_3d import inference

if __name__ == '__main__':
    inference(
            '/data/medical/cardiac/cta2mbf/data_140_20210602/5.mbf_myocardium', 
            '/data/medical/cardiac/cta2mbf/data_140_20210602/6.inference_352x352x160_train', 
            '/home/zhangwd/code/work/MedCommon/gan/unit_test/checkpoints/bk/train_latest/1140_net_G.pth'
        )
