import os
from data import recdata


class my_reconstruction(recdata.RecData):
    def __init__(self, args, name='my_reconstruction', train=False, benchmark=False):
        super(my_reconstruction, self).__init__(
            args, name=name, train=train, benchmark=benchmark)

    def _set_filesystem(self, data_dir):
        super(my_reconstruction, self)._set_filesystem(data_dir)
        self.dir_unpaired = os.path.join(data_dir, 'test') # 添加一个未知unpair目录