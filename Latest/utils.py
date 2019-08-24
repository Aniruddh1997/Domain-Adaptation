import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

def free_params(module):
    for p in module.parameters():
        p.requires_grad = True


def frozen_params(module):
    for p in module.parameters():
        p.requires_grad = False

class VGG_Encoder(nn.Module):
    def __init__(self, args):
        super(VGG_Encoder, self).__init__()

        self.n_z = 3 * args.n_z // 4
        self.conv = nn.Conv2d(512, self.n_z, 2, 1, 0, bias=False)

    def forward(self, x):
        x = self.conv(x)
        return x.squeeze()

class Encoder64(nn.Module):
    def __init__(self, args):
        super(Encoder64, self).__init__()

        self.n_channel = args.n_channel
        self.dim_h = args.dim_h
        self.n_z = args.n_z 

        self.main = nn.Sequential(
            nn.Conv2d(self.n_channel, self.dim_h, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(self.dim_h, self.dim_h * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),
            nn.Conv2d(self.dim_h * 2, self.dim_h * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(True),
            nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 8),
            nn.ReLU(True),
            nn.Conv2d(self.dim_h * 8, self.n_z, 4, 1, 0, bias=False) ################ line added ############
        )
        # self.fc = nn.Linear(self.dim_h * (2 ** 3), self.n_z)

    def forward(self, x):
        noise = Variable(0.25*torch.randn(x.shape).cuda())
        x = self.main(x + noise)
        x = x.squeeze()
        # x = self.fc(x)
        return x

class Encoder32(nn.Module):
    def __init__(self, args):
        super(Encoder32, self).__init__()

        self.n_channel = args.n_channel
        self.dim_h = args.dim_h
        self.n_z = args.n_z 

        self.main = nn.Sequential(
            nn.Conv2d(self.n_channel, self.dim_h, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(self.dim_h, self.dim_h * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),
            nn.Conv2d(self.dim_h * 2, self.dim_h * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(True),
            nn.Conv2d(self.dim_h * 4, self.n_z, 4, 1, 0, bias=False) ################ line added ############
        )
        # self.fc = nn.Linear(self.dim_h * (2 ** 3), self.n_z)

    def forward(self, x):
        noise = Variable(0.1*torch.randn(x.shape).cuda())
        x = self.main(x + noise)
        x = x.squeeze()
        # x = self.fc(x)
        return x

# changed final activation to tanh from sigmoid in both decoder
class Decoder64(nn.Module):
    def __init__(self, args):
        super(Decoder64, self).__init__()

        self.n_channel = args.n_channel
        self.dim_h = args.dim_h
        self.n_z = args.n_z

        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.n_z, self.dim_h * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.dim_h * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 8, self.dim_h * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 4, self.dim_h * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 2, self.dim_h, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h, self.n_channel, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.n_z, 1, 1)
        x = self.main(x)
        return x

class Decoder32(nn.Module):
    def __init__(self, args):
        super(Decoder32, self).__init__()

        self.n_channel = args.n_channel
        self.dim_h = args.dim_h
        self.n_z = args.n_z

        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.n_z, self.dim_h * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 4, self.dim_h * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 2, self.dim_h, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h, self.n_channel, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.n_z, 1, 1)
        x = self.main(x)
        return x


def imq_kernel(X: torch.Tensor,
               Y: torch.Tensor,
               h_dim: int):
    batch_size = X.size(0)

    norms_x = X.pow(2).sum(dim = 1, keepdim = True)
    prods_x = torch.mm(X, X.t())
    dists_x = norms_x + norms_x.t() - 2 * prods_x

    norms_y = Y.pow(2).sum(dim = 1, keepdim = True)
    prods_y = torch.mm(Y, Y.t())
    dists_y = norms_y + norms_y.t() - 2 * prods_y

    prods_xy = torch.mm(X, Y.t())
    dists_c = norms_x + norms_y.t() - 2 * prods_xy

    stats = 0
    for scale in [.1, .2, .5, 1., 2., 5., 10.]:
        C = 2 * h_dim * 1.0 * scale
        res1 = C / (C + dists_x)
        res1 += C / (C + dists_y)

        if torch.cuda.is_available():
            res1 = (1 - torch.eye(batch_size).cuda()) * res1
        else:
            res1 = (1 - torch.eye(batch_size)) * res1

        res1 = res1.sum() / (batch_size * batch_size - batch_size)
        res2 = C / (C + dists_c)
        res2 = res2.sum() * 2. / (batch_size * batch_size)
        stats += res1 - res2

    return stats


def rbf_kernel(X: torch.Tensor,
               Y: torch.Tensor,
               h_dim: int):
    batch_size = X.size(0)

    p2_norm_x = X.pow(2).sum(1).unsqueeze(0)
    norms_x = X.sum(1).unsqueeze(0)
    prods_x = torch.mm(norms_x, norms_x.t())
    dists_x = p2_norm_x + p2_norm_x.t() - 2 * prods_x

    p2_norm_y = Y.pow(2).sum(1).unsqueeze(0)
    norms_y = X.sum(1).unsqueeze(0)
    prods_y = torch.mm(norms_y, norms_y.t())
    dists_y = p2_norm_y + p2_norm_y.t() - 2 * prods_y

    dot_prd = torch.mm(norms_x, norms_y.t())
    dists_c = p2_norm_x + p2_norm_y.t() - 2 * dot_prd

    stats = 0
    for scale in [.1, .2, .5, 1., 2., 5., 10.]:
        C = 2 * h_dim * 1.0 / scale
        res1 = torch.exp(-C * dists_x)
        res1 += torch.exp(-C * dists_y)

        if torch.cuda.is_available():
            res1 = (1 - torch.eye(batch_size).cuda()) * res1
        else:
            res1 = (1 - torch.eye(batch_size)) * res1

        res1 = res1.sum() / (batch_size - 1)
        res2 = torch.exp(-C * dists_c)
        res2 = res2.sum() * 2. / batch_size
        stats += res1 - res2

    return stats

def get_data(path):
    data_np = np.load(path)
    data_np = data_np.transpose((0, 3, 1, 2))
    data_np = data_np / 255.0
    # data_np = (data_np - 0.5)/0.5
    return data_np

def generateTheta(L,endim):
    # This function generates L random samples from the unit `ndim'-u
        theta=torch.FloatTensor([w/np.sqrt((w**2).sum()) for w in np.random.normal(size=(L,endim))])
        if torch.cuda.is_available():
            theta = theta.cuda()
        return theta

def SW(q_samples,p_samples, thetas):
    #q_samples has size batchsize,endim
    proj_q = torch.mm(q_samples, thetas.t()) #batchsize x L
    proj_p = torch.mm(p_samples, thetas.t())
    proj_q_sorted, _ = torch.sort(proj_q, dim=0)
    proj_p_sorted, _ = torch.sort(proj_p, dim=0)
    err = (proj_q_sorted - proj_p_sorted)**2
    SW_dist = torch.mean(err)
    return SW_dist

def visualize(epoch, y_np, ims_dir, n_vis=64):
    n, nc, sz, _ = y_np.shape

    images_y = torch.FloatTensor(n, nc, sz, sz)
    images_y = Variable(images_y.cuda())
    images_y.data.copy_(torch.from_numpy(y_np))

    z_y = encoderY(images_y[:n_vis])
    z_y_common = z_y[:,50]
    z_x_private = torch.rand_like(z_y_common)
    z_x = torch.cat((z_y_common, z_x_private), 1)

    I_implicit = decoderX(z_x)
    I_target = Variable(torch.from_numpy(y_np[:n_vis]).float()).cuda()

    save_image(I_implicit, '%s/implicit_epoch_%03d.png' % (ims_dir, epoch))
    save_image(I_target, '%s/target.png' % ims_dir)

def format_im(images):
    images = images.clone()
    # images = images*0.5 + 0.5
    images = torch.clamp(images.data, 0, 1)
    return images

def prior(batch_size, n_z):
    res = Variable(torch.randn(batch_size, args.n_z))
    mean = np.array([-0.5, 0.5])
    component_id = np.random.choice(2, size=batch_size)
    for i in range(batch_size):
        res[i] += mean[component_id[i]]

    return res