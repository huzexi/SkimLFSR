'''
@ARTICLE{SkimLFSR,
  author={Hu, Zeke Zexi and Chen, Haodong and Ye, Hui and Chen, Xiaoming
          and Chung, Vera Yuk Ying and Shen, Yiran and Cai, Weidong},
  journal={IEEE Transactions on Multimedia},
  title={Less is More: Skim Transformer for Light Field Image Super-Resolution},
  year={2026},
  note={https://ieeexplore.ieee.org/document/11569350/}
}
'''
import torch.nn as nn

from .SkimLFSR import SkimLFSR as Network
from .SkimLFSR import SkimLFSRConfig as NetworkConfig


# For BasicLFSR
def get_model(args):
    config = NetworkConfig.from_scale(args.scale_factor)
    return Network(scale=args.scale_factor, sz_a=[args.angRes_in, args.angRes_in], config=config)


class get_loss(nn.Module):
    """L1 reconstruction loss, following the BasicLFSR training convention."""
    def __init__(self, args):
        super(get_loss, self).__init__()
        self.criterion_Loss = nn.L1Loss()

    def forward(self, SR, HR, data_info=None):
        return self.criterion_Loss(SR, HR)


def weights_init(m):
    # The network initializes its own weights in its submodules' constructors
    # (kaiming_uniform on the attention/linear layers), so no extra per-module
    # initialization is required here.
    pass
