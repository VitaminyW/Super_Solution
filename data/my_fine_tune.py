import os
from data import ftdata


class my_fine_tune(ftdata.FTData):
    def __init__(self, args, name='my_train', train=True, benchmark=False,unpaired_rate = 0.3):
        super(my_fine_tune, self).__init__(
            args, name=name, train=train, benchmark=benchmark,unpaired_rate=unpaired_rate
        )

    def _set_filesystem(self, data_dir):
        super(my_fine_tune, self)._set_filesystem(data_dir)
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
        self.dir_unpaired = os.path.join(data_dir, 'test') # 添加一个未知unpair目录