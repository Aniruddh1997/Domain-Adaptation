import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR

from utils import Encoder32, Decoder32, Encoder64, Decoder64, imq_kernel, rbf_kernel
from utils import get_data, generateTheta, SW
from utils import visualize, format_im
from perceptual_loss import _VGGDistance

torch.manual_seed(123)

parser = argparse.ArgumentParser(description='PyTorch MNIST WAE-MMD')
parser.add_argument('-batch_size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('-epochs', type=int, default=100, help='number of epochs to train (default: 500)')
parser.add_argument('-lr', type=float, default=1e-4, help='learning rate (default: 0.0001)')
parser.add_argument('-dim_h', type=int, default=32, help='hidden dimension (default: 32)')
parser.add_argument('-n_z', type=int, default=64, help='hidden dimension of z (default: 100)')
parser.add_argument('-LAMBDA', type=float, default=10, help='regularization coef MMD term (default: 10)')
parser.add_argument('-n_channel', type=int, default=3, help='input channels (default: 1)')
parser.add_argument('-sigma', type=float, default=1, help='variance of hidden dimension (default: 1)')

args = parser.parse_args()


X_images = get_data('data/edges2shoes_x.npy')
Y_images = get_data('data/edges2shoes_y.npy')

encoderX, decoderX = Encoder64(args), Decoder64(args)
encoderY, decoderY = Encoder64(args), Decoder64(args)

encoderX.load_state_dict(torch.load("nets/E_X.pth"))
decoderX.load_state_dict(torch.load("nets/D_X.pth"))
encoderY.load_state_dict(torch.load("nets/E_Y.pth"))
decoderY.load_state_dict(torch.load("nets/D_Y.pth"))

encoderX.eval()
decoderX.eval()
encoderY.eval()
decoderY.eval()

if torch.cuda.is_available():
    encoderX, decoderX = encoderX.cuda(), decoderX.cuda()
    encoderY, decoderY = encoderY.cuda(), decoderY.cuda()


one = torch.Tensor([1])
mone = one * -1

if torch.cuda.is_available():
    one = one.cuda()
    mone = mone.cuda()


nc = 3
sz = 64
ids = np.arange(8)
images = torch.FloatTensor(8, nc, sz, sz)
images = Variable(images.cuda())
images.copy_(torch.from_numpy(Y_images[ids]))
output_implicit = torch.FloatTensor(48,nc,sz,sz).cuda()

output_implicit[:8] = images
for i in range(5):
    z_y = encoderY(images)
    I_implicit = decoderX(z_y)
    output_implicit[8*(i+1):8*(i+2)] = I_implicit

ims = format_im(output_implicit)
save_image(ims, 'images/eval_out.png', normalize=False)
