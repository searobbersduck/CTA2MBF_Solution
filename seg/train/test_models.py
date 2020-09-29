import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, 'external_lib.MedicalZooPytorch'))
from external_lib.MedicalZooPytorch.lib.medzoo.Unet3D import UNet3D
from external_lib.MedicalZooPytorch.lib.losses3D.dice import DiceLoss

import torch
import torch.nn


model = UNet3D(1, 2)
model1 = torch.nn.DataParallel(model).cuda()

inp = torch.rand(2,1,128,128,128)
gt = np.random.randint(0,2,size=(2,128,128,128))
gt = torch.from_numpy(gt).long().cuda()
output = model1(inp.cuda())

criterion = DiceLoss(2)

loss = criterion(output, gt)

print(loss)

print('hello world!')



