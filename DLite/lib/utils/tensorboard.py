import pprint
import re
import math
from torch.utils.tensorboard import SummaryWriter
import torchvision

class TBWriter(object):
    def __init__(self, tb_path, args):
        self.writer = SummaryWriter(tb_path)
        self.writer.add_text('config', re.sub("\n", "  \n", pprint.pformat(args, width=1)), 0)
        self.writer.flush()
    
    def add_scalars(self, tag, value, global_step):
        self.writer.add_scalar(tag, value, global_step=global_step)
    
    def add_image(self, tag, img, global_step):
        B = img.shape[0]
        nrow = math.ceil(B/math.floor(math.sqrt(B)))
        img = torchvision.utils.make_grid(img, nrow=nrow, normalize=True, scale_each=True)
        self.writer.add_image(tag, img, global_step=global_step)
