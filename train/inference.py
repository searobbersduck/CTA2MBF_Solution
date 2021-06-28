import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
print(ROOT)

sys.path.append(ROOT)

from external_lib.MedCommon.gan.runner.train_gan_3d import inference, GANTrainer

if __name__ == '__main__':
    # inference(
    #         '/data/medical/cardiac/cta2mbf/data_140_20210602/5.mbf_myocardium', 
    #         '/data/medical/cardiac/cta2mbf/data_140_20210602/6.inference_352x352x160_train', 
    #         '/home/zhangwd/code/work/MedCommon/gan/unit_test/checkpoints/bk/train_latest/1140_net_G.pth'
    #     )
    # # 生成2D图片
    GANTrainer.export_slicemap_multiprocessing(
            '/data/medical/cardiac/cta2mbf/data_66_20210517/6.inference_384x384x160_eval', 
            '/data/medical/cardiac/cta2mbf/data_66_20210517/6.inference_384x384x160_eval_slicemap', 
            src_ww=150, src_wl=75, dst_ww=200, dst_wl=100, src_lut=None, dst_lut='jet'
        )
    # # 统计MAE指标
    # GANTrainer.calc_mae(
    #         data_root='/data/medical/cardiac/cta2mbf/data_66_20210517/6.inference_384x384x160_eval', 
    #         out_dir = '/data/medical/cardiac/cta2mbf/data_66_20210517/7.analysis_result', 
    #         out_file = 'mae_384x384x160_eval.csv'
    #     )
