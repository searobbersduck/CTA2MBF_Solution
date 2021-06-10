import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
print(ROOT)

sys.path.append(ROOT)

from external_lib.MedCommon.gan.runner.train_gan_3d import train

if __name__ == '__main__':
    train()


