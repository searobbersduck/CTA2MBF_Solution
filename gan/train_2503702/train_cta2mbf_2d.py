import os
import sys

curdir = os.path.abspath(os.curdir) # $Lung_COPD_PATH/models/2d
root_dir=os.path.join(os.path.dirname(os.path.dirname(curdir)))
sys.path.append(os.path.dirname(curdir))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'external_lib/pytorch-CycleGAN-and-pix2pix'))

from models.pix2pix_model import Pix2PixModel
from options.train_options import TrainOptions
from options.test_options import TestOptions
from util.visualizer import save_images
from util import html

import time
from models import create_model
from util.visualizer import Visualizer

from datasets.slice.cta2bf_datasets import CTA2MBF_DS
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import SimpleITK as sitk
import torch
import numpy as np

def train():
    opt = TrainOptions().parse()

    root_dirs = ['../../data/task0/2.slice_2d/train_2503702']
    config_xxx_files = ['../../data/task0/2.slice_2d/config/config_2d_cta2mbf_train_2503702.txt']
    config_yyy_files = ['../../data/task0/2.slice_2d/config/config_2d_cta2mbf_train_2503702.txt']
    crop_size = [512, 512]
    ds = CTA2MBF_DS(root_dirs, config_xxx_files, config_yyy_files, 'train', crop_size, crop_size)
    data_loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=16, pin_memory=True)

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)

    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        for i, (srcs, dsts, _, _) in tqdm(enumerate(data_loader)):
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            data = {}
            data['A'] = srcs
            data['B'] = dsts
            data['A_paths'] = 'A'
            data['B_paths'] = ['B']

            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / len(data_loader), losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
            
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

    print('hello world!')


def test():
    

    root_dirs = ['../../data/toy_data/2.slice_2d/val']
    config_xxx_files = ['../../data/toy_data/2.slice_2d/config/config_2d_copd_xxx_val.txt']
    config_yyy_files = ['../../data/toy_data/2.slice_2d/config/config_2d_copd_yyy_val.txt']
    crop_size = [512, 512]
    ds = COPD_GAN_DS(root_dirs, config_xxx_files, config_yyy_files, 'train', crop_size, crop_size)
    data_loader = DataLoader(ds, batch_size=12, shuffle=True, num_workers=16, pin_memory=True)

    model = create_model(opt)
    model.setup(opt)
    if opt.eval:
        model.eval()
    

def predict_onecase(infile_A, infile_B, outdir, model=None):
    '''
    debug cmd: predict_onecase('../../data/task2/1.1.raw/1.2.156.112605.14038007945377.191013010825.3.5228.61295_1.2.156.112605.14038007945377.191013011003.3.5228.104694/m_ptrRawImage.nii.gz', '../../data/task2/1.1.raw/1.2.156.112605.14038007945377.191013010825.3.5228.61295_1.2.156.112605.14038007945377.191013011003.3.5228.104694/diff.nii.gz', '../../data/tmp/1.nii.gz')
    '''
    if model is None:
        opt = TestOptions().parse()
        model = create_model(opt)
        model.setup(opt)
        model.eval()

    series_uid = os.path.basename(os.path.dirname(infile_A))
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch), '{}'.format(series_uid))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    os.makedirs(outdir, exist_ok=True)
    in_img_A = sitk.ReadImage(infile_A)
    in_arr_A = sitk.GetArrayFromImage(in_img_A)#h*w*c
    in_arr_A = in_arr_A.transpose((2,0,1))
    print(in_arr_A.shape)
    in_img_B = sitk.ReadImage(infile_B)
    in_arr_B = sitk.GetArrayFromImage(in_img_B)
    in_arr_B = in_arr_B.transpose((2,0,1))
    print(in_arr_B.shape)
    
    out_arr_B = np.zeros(in_arr_B.shape)

    assert in_arr_A.shape == in_arr_B.shape

    for i in range(in_arr_A.shape[0]):#C*H*W
        slice_arr_A = in_arr_A[i]
        slice_arr_B = in_arr_B[i]
        srcs = torch.from_numpy(slice_arr_A).unsqueeze(0).unsqueeze(0).float()
        dsts = torch.from_numpy(slice_arr_B).unsqueeze(0).unsqueeze(0).float()
        data = {}
        data['A'] = srcs
        data['B'] = dsts
        data['A_paths'] = ['{}_{}_A'.format(2835191-1, i)]
        data['B_paths'] = ['{}_{}_B'.format(2835191-1, i)]
        model.set_input(data)
        model.test()
        fake_B = model.fake_B.detach().cpu().numpy().squeeze()
        print(fake_B.shape)
        out_arr_B[i] = fake_B
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    webpage.save()
    out_arr_B = np.array(out_arr_B, dtype=np.int16)#c*h*w
    out_arr_B = out_arr_B.transpose((1,2,0))
    print(out_arr_B.shape)
    out_img_B = sitk.GetImageFromArray(out_arr_B)
    in_img_B.CopyInformation(in_img_A)
    out_img_B.CopyInformation(in_img_A)
    sitk.WriteImage(out_img_B, os.path.join(outdir, '2835191_use_2503702_fake_B.nii.gz'))
    sitk.WriteImage(in_img_A, os.path.join(outdir, '2835191_use_2503702_real_A.nii.gz'))
    sitk.WriteImage(in_img_B, os.path.join(outdir, '2835191_use_2503702_real_B.nii.gz'))

def predict_singletask(indir_2d, indir, outdir):
    '''
    predict_singletask('../../data/task2/2.slice_2d/val', '../../data/task2/1.1.raw', '../../data/task2/result1')
    '''
    for series_uid in os.listdir(indir_2d):
        series_path = os.path.join(indir, series_uid)
        file_a_path = os.path.join(series_path, 'm_ptrRawImage.nii.gz')
        # file_a_path = os.path.join(series_path, 'LungReg.MHA.Matched.nii.gz')
        file_b_path = os.path.join(series_path, 'diff.nii.gz')
        if not os.path.isfile(file_a_path):
            print('{} not exist'.format(file_a_path))
            continue
        if not os.path.isfile(file_b_path):
            print('{} not exist'.format(file_b_path))
            continue
        sub_out_dir = os.path.join(outdir, series_uid)
        predict_onecase(file_a_path, file_b_path, sub_out_dir)
        

if __name__ == '__main__':
    # train()
    # predict_onecase('../../data/task2/1.1.raw/1.2.156.112605.14038007945377.191013010825.3.5228.61295_1.2.156.112605.14038007945377.191013011003.3.5228.104694/m_ptrRawImage.nii.gz', '../../data/task2/1.1.raw/1.2.156.112605.14038007945377.191013010825.3.5228.61295_1.2.156.112605.14038007945377.191013011003.3.5228.104694/diff.nii.gz', '../../data/tmp/1.nii.gz')
    # predict_singletask('../../data/task2/2.slice_2d/val', '../../data/task2/1.1.raw', '../../data/task2/result2')
    # predict_singletask('../../data/task_suhai2/2.slice_2d/val', '../../data/task_suhai2/1.1.raw', '../../data/task_suhai2/result2')
    predict_onecase('/home/proxima-sx12/data/after_registration/CTA_MBF_usefulnii/2835191_cta_141.nii.gz', '/home/proxima-sx12/data/after_registration/CTA_MBF_usefulnii/2835191_mbf_141.nii.gz', '/home/proxima-sx12/data/gnerated_results/')