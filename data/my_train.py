import os
from data import srdata


class my_train(srdata.SRData):
    def __init__(self, args, name='my_train', train=True, benchmark=False):
        super(my_train, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _set_filesystem(self, data_dir):
        super(my_train, self)._set_filesystem(data_dir)
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic')