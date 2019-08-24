# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch
import opt_nam
import vaeNAM
import waeNAM
import utils
import params
import sys

if sys.argv[2] == "mnist":
	params.x_data_path = "../../Latest/data/mnist.npy"
	params.y_data_path = "../../Latest/data/svhn.npy"
	params.uncon_shape = (32,32,1)
	params.uncon_cp_path = "unconditional_nets/netG_25_mnist.pth"
	params.nz = 16
elif sys.argv[2] == "svhn":
	params.x_data_path = "../../Latest/data/svhn.npy"
	params.y_data_path = "../../Latest/data/mnist.npy"
	params.uncon_shape = (32,32,3)
	params.uncon_cp_path = "unconditional_nets/netG_25_svhn.pth"
	params.nz = 64


# Load pretrained unconditional generator for X domain.
def get_x_generator():
    netG = utils.generator(params.uncon_shape[0],
                           params.nz,
                           params.uncon_ngf,
                           params.uncon_shape[2])
    state_dict = torch.load(params.uncon_cp_path)
    netG.load_state_dict(state_dict)
    return netG


netG = get_x_generator()
netG.cuda()


# load Y domain data. Assumed to be a uint8 numpy array of shape (n, sz, sz, nc)
def get_y_data():
    data_np = np.load(params.y_data_path)
    data_np = data_np.transpose((0, 3, 1, 2))
    data_np = data_np / 255.0
    return data_np


y = get_y_data()
for i in range(y.shape[1]):
    y[:, i:i+1] -= params.mu[i]
    y[:, i:i+1] /= params.sd[i]
rp = np.random.permutation(y.shape[0])[:params.n_examples]
y = y[rp]


net_params = utils.NAMParams(nz=params.nz,
                             ngf=params.nam_ngf,
                             ndf=params.nam_ndf,
                             mu=params.mu,
                             sd=params.sd,
                             force_l2=False)
opt_params = utils.OptParams(lr=params.lr,
                             batch_size=params.batch_size,
                             epochs=params.epochs,
                             decay_epochs=params.decay_epochs,
                             decay_rate=params.decay_rate,
                             lr_ratio=params.lr_ratio)


if sys.argv[1] == "nam":
	nm = opt_nam.NAMTrainer(netG, params.uncon_shape, net_params)
elif sys.argv[1] == "vae":
	nm = vaeNAM.vaeNAMTrainer(netG, params.uncon_shape, net_params)
elif sys.argv[1] == "wae":
	nm = waeNAM.waeNAMTrainer(netG, params.uncon_shape, net_params)

nm.train_nam(y, opt_params)
