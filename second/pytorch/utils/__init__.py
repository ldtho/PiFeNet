import time
import contextlib
import torch
# from .dcn import DeformableConv2d
from .position_encoding import PositionEmbeddingSine, PositionEmbeddingLearned, build_position_encoding
from .separable_conv import SeparableConvBlock
from .swish import SwishImplementation, Swish, MemoryEfficientSwish
from .mlp import MLP
from .misc import inverse_sigmoid


@contextlib.contextmanager
def torch_timer(name=''):
    torch.cuda.synchronize()
    t = time.time()
    yield
    torch.cuda.synchronize()
    print(name, "time:", time.time() - t)



#lyft-dataset-sdk
