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
criterion_VGG5 = _VGGDistance(5)
criterion_VGG3 = _VGGDistance(3)

encoderX.train()
decoderX.train()
encoderY.train()
decoderY.train()

if torch.cuda.is_available():
    encoderX, decoderX = encoderX.cuda(), decoderX.cuda()
    encoderY, decoderY = encoderY.cuda(), decoderY.cuda()


one = torch.Tensor([1])
mone = one * -1

if torch.cuda.is_available():
    one = one.cuda()
    mone = mone.cuda()

# Optimizers
optim_params = list(encoderX.parameters()) + list(decoderX.parameters()) + \
               list(encoderY.parameters())  + list(decoderY.parameters())
               
optim = optim.Adam(optim_params, lr=args.lr)

for epoch in range(args.epochs):
    
    step = 0
    beta = (1 - np.exp(-epoch/20))
    beta_cross = 0.6*np.exp(-epoch/2)

    n_x, nc_x, sz, _ = X_images.shape
    n_y, nc_y, sz, _ = Y_images.shape
    rp_x = np.random.permutation(n_x)
    ny = []
    for j in range(n_x//n_y):
        ny.extend(np.arange(n_y))
    ny.extend(np.arange(n_x % n_y))
    rp_y = np.random.permutation(ny)
    # Compute batch size
    batch_size = args.batch_size
    batch_n = n_x // batch_size
    # Initialize tensors
    X_batch = Variable(torch.FloatTensor(batch_size, nc_x, sz, sz))
    Y_batch = Variable(torch.FloatTensor(batch_size, nc_y, sz, sz))

    if torch.cuda.is_available():
        X_batch = X_batch.cuda()
        Y_batch = Y_batch.cuda()


    loss_avg = 0
    recon_avg_x = 0
    recon_avg_y = 0
    cross_x_avg = 0
    cross_y_avg = 0
    mmd_avg = 0

    for i in tqdm(range(batch_n)):
        # Load the shuffled data 
        np_data_x = X_images[rp_x[i * batch_size: (i + 1) * batch_size]]
        np_data_y = Y_images[rp_y[i * batch_size: (i + 1) * batch_size]]
        X_batch.data.copy_(torch.from_numpy(np_data_x))
        Y_batch.data.copy_(torch.from_numpy(np_data_y))
 
        # ======== Train Generator ======== #

        z_x = encoderX(X_batch)
        z_y = encoderY(Y_batch)
        x_recon = decoderX(z_x)
        y_recon = decoderY(z_y)
        

        recon_loss_x = criterion_VGG5(X_batch, x_recon)
        recon_loss_y = criterion_VGG5(Y_batch, y_recon)

        # ======== MMD Kernel Loss ======== #

        z_fake = Variable(torch.randn(batch_size, args.n_z) * args.sigma)
        if torch.cuda.is_available():
            z_fake = z_fake.cuda()

        
        mmd_loss = imq_kernel(z_x, z_y, encoderX.n_z)
        cross_term_x = criterion_VGG3(x_recon, decoderY(z_x))
        cross_term_y = criterion_VGG3(y_recon, decoderX(z_y))

        optim.zero_grad()
        total_loss = 5*(recon_loss_x + recon_loss_y) + beta*(mmd_loss) + beta_cross*(cross_term_x + cross_term_y)
        total_loss.backward()
        optim.step()

        loss_avg += total_loss.item() / batch_n
        recon_avg_x += recon_loss_x.item() / batch_n
        recon_avg_y += recon_loss_y.item() / batch_n
        cross_x_avg += cross_term_x.item() / batch_n
        cross_y_avg += cross_term_y.item() / batch_n
        mmd_avg += mmd_loss.item() / batch_n

        step += 1

    print("Epoch: [%d/%d],  Reconstruction Loss: %.4f %.4f MMD Loss : %.4f Cross Term : %.4f %.4f Total Loss : %.4f"  % 
        (epoch + 1, args.epochs, recon_avg_x, recon_avg_y, mmd_avg, cross_x_avg, cross_y_avg, loss_avg))


    if epoch % 10 == 0 :
        images_y = torch.FloatTensor(64, nc_y, sz, sz)
        images_y = Variable(images_y.cuda())
        images_y.data.copy_(torch.from_numpy(Y_images[:64]))

        with torch.no_grad():
            z_y = encoderY(images_y)
            I_implicit = decoderX(z_y)
            I_target_est = decoderY(z_y)
            I_target = Variable(torch.from_numpy(Y_images[:64]).float()).cuda()
            
            I_target_est = format_im(I_target_est)
            I_implicit = format_im(I_implicit)
            I_target = format_im(I_target)

            save_image(I_implicit, 'images/implicit_epoch_%03d.png' % epoch)
            save_image(I_target_est, 'images/target_est_%03d.png' % epoch)
            save_image(I_target, 'images/target.png')

        torch.save(encoderX.state_dict(), 'nets/E_X.pth')
        torch.save(decoderX.state_dict(), 'nets/D_X.pth')
        torch.save(encoderY.state_dict(), 'nets/E_Y.pth')
        torch.save(decoderY.state_dict(), 'nets/D_Y.pth')
    