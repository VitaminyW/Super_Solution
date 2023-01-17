import os
import glob
from data import common
import numpy as np
import imageio
import torch.utils.data as data
import torch


class FTData(data.Dataset):
    """
    用于使用低分辨率图片进行fine-tune
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
        lrs, hrs, filenames, unpairs = [], [], [], []
        for idx in idxs:
            if np.random.rand() > self.unpaired_rate:  # 生成配对的数据
                lr, hr, filename = self._load_file(idx)
                lr, hr = self.get_patch(lr, hr)
                lr, hr = common.set_channel(lr, hr, n_channels=self.args.n_colors)
                lr_tensor, hr_tensor = common.np2Tensor(
                    lr, hr, rgb_range=self.args.rgb_range
                )
                lrs.append(lr_tensor)
                hrs.append(hr_tensor)
                filenames.append(filename)
            else:
                idx = self._get_index(idx, True)
                f_unpaired = self.images_unpaired[idx]
                unpaired = imageio.imread(f_unpaired)
                filename, _ = os.path.splitext(os.path.basename(f_unpaired))
                unpaired = self.get_patch_unpaired(unpaired)
                unpaired = common.set_channel(unpaired, n_channels=self.args.n_colors, unpaired=True)
                unpaired_tensor = common.np2Tensor(
                    unpaired, rgb_range=self.args.rgb_range, unpaired=True
                )
                unpairs.append(unpaired_tensor)
        if len(lrs) > 0:  # lrs可能等于0
            scales = len(lrs[0])
            lrs_temp = []
            for scale_index in range(scales):
                lrs_temp.append([])
            for item in lrs:
                for scale_index in range(scales):
                    lrs_temp[scale_index].append(item[scale_index])
            for scale_index in range(scales):
                lrs_temp[scale_index] = torch.stack(lrs_temp[scale_index], dim=0)
            lrs = lrs_temp
        return lrs, torch.stack(hrs) if len(hrs) > 0 else hrs, filenames, torch.stack(unpairs, dim=0) if len(
            unpairs) > 0 else unpairs

    def __len__(self):
        return self.dataset_length

    def _get_imgs_path(self, args):
        list_hr, list_lr, list_unpaired = self._scan()
        self.images_hr, self.images_lr, self.images_unpaired = list_hr, list_lr, list_unpaired

    def _set_dataset_length(self):
        if self.train:
            self.dataset_length = self.args.test_every * self.args.batch_size
            # print(self.dataset_length)
            repeat = self.dataset_length // (len(self.images_hr) + len(self.images_unpaired))
            print(f'the repeat time of train_dataset is {repeat}')
            print(f'len(self.images_hr):{len(self.images_hr)}')
            print(f'len(self.images_unpaired):{len(self.images_unpaired)}')
            self.random_border = len(self.images_hr) * repeat
        else:
            self.dataset_length = len(self.images_hr)

    def _scan(self):
        """
        获取各个放缩倍速下的图片路径和高分辨率图片
        :return:
        """
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0]))
        )
        names_lr = [[] for _ in self.scale]
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            for si, s in enumerate(self.scale):
                names_lr[si].append(os.path.join(
                    self.dir_lr, 'X{}/{}x{}{}'.format(
                        s, filename, s, self.ext[1]
                    )
                ))
        names_unparied = sorted(glob.glob(os.path.join(self.dir_unpaired, '*' + self.ext[0])))
        return names_hr, names_lr, names_unparied

    def _set_filesystem(self, data_dir):
        self.apath = os.path.join(data_dir, self.name)
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
        self.dir_unpaired = os.path.join(data_dir, 'test')  # 添加一个未知unpair目录
        self.ext = ('.png', '.png')

    def _get_index(self, idx, unpaired):
        if self.train:
            if not unpaired:
                return np.random.randint(len(self.images_hr))  # 随机返回
            else:
                return np.random.randint(len(self.images_unpaired))  # 随机返回
        else:
            return idx

    def _load_file(self, idx):
        idx = self._get_index(idx, False)
        f_hr = self.images_hr[idx]
        f_lr = [self.images_lr[idx_scale][idx] for idx_scale in range(len(self.scale))]
        filename, _ = os.path.splitext(os.path.basename(f_hr))
        hr = imageio.imread(f_hr)
        lr = [imageio.imread(f_lr[idx_scale]) for idx_scale in range(len(self.scale))]
        return lr, hr, filename

    def get_patch(self, lr, hr):
        scale = self.scale
        multi_scale = len(self.scale) > 1
        if self.train:
            lr, hr = common.get_patch(
                lr,
                hr,
                patch_size=self.args.patch_size,
                scale=scale,
                multi_scale=multi_scale
            )
            if not self.args.no_augment:
                lr, hr = common.augment(lr, hr)
        else:
            if isinstance(lr, list):
                ih, iw = lr[0].shape[:2]
            else:
                ih, iw = lr.shape[:2]
            hr = hr[0:ih * scale[0], 0:iw * scale[0]]

        return lr, hr

    def get_patch_unpaired(self, unpaired):
        scale = self.scale[0]
        if self.train:
            lr = common.get_patch_unpaired(
                unpaired,
                patch_size=self.args.patch_size,
                scale=scale
            )
            if not self.args.no_augment:
                lr = common.augment(lr, unpaired=True)
        else:
            ih, iw = unpaired.shape[:2]
            # hr = hr[0:ih * scale[0], 0:iw * scale[0]]
        return lr
