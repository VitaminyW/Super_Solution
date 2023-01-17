import os
import glob
from data import common
import numpy as np
import imageio
import torch.utils.data as data
import torch


class RecData(data.Dataset):
    """
    用于使用低分辨率图片进行重建
    """

    def __init__(self, args, name='', train=True, benchmark=False, unpaired_rate=0.3):
        self.args = args
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'
        self.do_eval = True
        self.benchmark = benchmark
        self.scale = args.scale.copy()
        self.scale.reverse()
        self.unpaired_rate = unpaired_rate
        self._set_filesystem(args.data_dir)
        self._get_imgs_path(args)
        self._set_dataset_length()

    def __getitems__(self, idxs):
        unpairs = []
        for idx in idxs:
            f_unpaired = self.images_unpaired[idx]
            unpaired = imageio.imread(f_unpaired)
            filename, _ = os.path.splitext(os.path.basename(f_unpaired))
            unpaired = common.set_channel(unpaired, n_channels=self.args.n_colors, unpaired=True)
            unpaired_tensor = common.np2Tensor(
                unpaired, rgb_range=self.args.rgb_range, unpaired=True
            )
            unpairs.append(unpaired_tensor)
            filename, _ = os.path.splitext(os.path.basename(f_unpaired))
        return torch.stack(unpairs, dim=0),filename

    def __len__(self):
        return self.dataset_length

    def _get_imgs_path(self, args):
        list_unpaired = self._scan()
        self.images_unpaired = list_unpaired

    def _set_dataset_length(self):
        self.dataset_length = len(self.images_unpaired)

    def _scan(self):
        """
        获取各个放缩倍速下的图片路径和高分辨率图片
        :return:
        """
        names_unparied = sorted(glob.glob(os.path.join(self.dir_unpaired, '*' + self.ext[0])))
        return names_unparied

    def _set_filesystem(self, data_dir):
        self.dir_unpaired = os.path.join(data_dir, 'test')  # 添加一个unpair目录
        self.ext = ('.png', '.png')
