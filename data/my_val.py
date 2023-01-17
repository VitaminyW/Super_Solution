import os
from data import srdata


class my_val(srdata.SRData):
    def __init__(self, args, name='my_val', train=True, benchmark=False):
        super(my_val, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _set_filesystem(self, data_dir):
        super(my_val, self)._set_filesystem(data_dir)
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic')