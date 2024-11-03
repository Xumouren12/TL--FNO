import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial
from timeit import default_timer
from utilities3 import *
import scipy.io as scio
from Adam import Adam

def setup_sedd(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
setup_sedd(0)


class DataSet:
    def __init__(self, bs):
        self.bs = bs
        self.F_train, self.Ux_train, self.Uy_train, self.F_test, self.Ux_test, self.Uy_test, \
        self.X, self.ux_mean, self.ux_std, self.uy_mean, self.uy_std = self.load_data()


    def decoder(self, x, y):
        x = (x - 8.5) * (self.ux_std + 1.0e-9) + self.ux_mean
        y = (y - 8.5) * (self.uy_std + 1.0e-9) + self.uy_mean
        x = x * 1e-5
        y = y * 1e-5
        return x, y


    def load_data(self):
        # Source data
        file = scio.loadmat('Dataset_1Circle.mat')
        s_bc = 101
        s = 1020
        f_train = file['f_bc_train']
        ux_train = file['ux_train'] * 1e5
        uy_train = file['uy_train'] * 1e5

        f_test = file['f_bc_test']
        ux_test = file['ux_test'] * 1e5
        uy_test = file['uy_test'] * 1e5

        xx = file['xx']
        yy = file['yy']
        xx = np.reshape(xx, (-1, 1))
        yy = np.reshape(yy, (-1, 1))
        X = np.hstack((xx, yy))

        f_train_mean = np.mean(np.reshape(f_train, (-1, s_bc)), 0)
        f_train_std = np.std(np.reshape(f_train, (-1, s_bc)), 0)
        ux_train_mean = np.mean(np.reshape(ux_train, (-1, s)), 0)
        ux_train_std = np.std(np.reshape(ux_train, (-1, s)), 0)
        uy_train_mean = np.mean(np.reshape(uy_train, (-1, s)), 0)
        uy_train_std = np.std(np.reshape(uy_train, (-1, s)), 0)

        f_train_mean = np.reshape(f_train_mean, (-1, 1, s_bc))
        f_train_std = np.reshape(f_train_std, (-1, 1, s_bc))
        ux_train_mean = np.reshape(ux_train_mean, (-1, s, 1))
        ux_train_std = np.reshape(ux_train_std, (-1, s, 1))
        uy_train_mean = np.reshape(uy_train_mean, (-1, s, 1))
        uy_train_std = np.reshape(uy_train_std, (-1, s, 1))

        F_train = np.reshape(f_train, (-1, 1, s_bc))
        F_train = (F_train - f_train_mean) / (f_train_std + 1.0e-9)
        Ux_train = np.reshape(ux_train, (-1, s, 1))
        Ux_train = (Ux_train - ux_train_mean) / (ux_train_std + 1.0e-9) + 8.5
        Uy_train = np.reshape(uy_train, (-1, s, 1))
        Uy_train = (Uy_train - uy_train_mean) / (uy_train_std + 1.0e-9) + 8.5

        F_test = np.reshape(f_test, (-1, 1, s_bc))
        F_test = (F_test - f_train_mean) / (f_train_std + 1.0e-9)
        Ux_test = np.reshape(ux_test, (-1, s, 1))
        Ux_test = (Ux_test - ux_train_mean) / (ux_train_std + 1.0e-9) + 8.5
        Uy_test = np.reshape(uy_test, (-1, s, 1))
        Uy_test = (Uy_test - uy_train_mean) / (uy_train_std + 1.0e-9) + 8.5

        return F_train, Ux_train, Uy_train, F_test, Ux_test, Uy_test, X, ux_train_mean, \
               ux_train_std, uy_train_mean, uy_train_std

################################################################
#  1d fourier layer
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FNO1d(nn.Module):
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.padding = 2  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(2, self.width)  # input channel is 2: (a(x), x)

        self.conv0 = SpectralConv1d(self.width, self.width, 20)
        self.conv1 = SpectralConv1d(self.width, self.width, 20)
        self.conv2 = SpectralConv1d(self.width, self.width, 16)
        self.conv3 = SpectralConv1d(self.width, self.width, 16)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.w4 = nn.Conv1d(self.width, self.width, 2, stride=2)
        self.w5 = nn.Conv1d(self.width, self.width, 2, stride=2)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 2)

        self.fc3 = nn.Linear(101, 1020)

        dict = torch.load("D:\Transfermodels\TL5_source_3_matrix")

        self.lambda0 = dict["lambda0"]
        self.lambda2 = dict["lambda2"]
        self.lambda4 = dict["lambda4"]
        self.lambda5 = dict["lambda5"]
        self.lambda7 = dict["lambda7"]
        self.lambda9 = dict["lambda9"]


    def forward(self, x):
        x = x.to(torch.float32)
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        # x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x_f1 = x1 + x2

        x_f1_all = x_f1 + x * self.lambda0
        x_f1_all = F.gelu(x_f1_all)

        x1 = self.conv1(x_f1_all)
        x2 = self.w1(x_f1_all)
        x_f2 = x1 + x2

        x_f2_all = x_f2 + x * self.lambda2
        x_f2_all = F.gelu(x_f2_all)

        x1 = self.conv2(x_f2_all)
        x2 = self.w2(x_f2_all)
        x_f3 = x1 + x2

        # x_f3_all = x_f3 + x
        x_f3_all = x_f3 + x_f1 * self.lambda4 + x * self.lambda5
        x_f3_all = F.gelu(x_f3_all)

        x1 = self.conv3(x_f3_all)
        x2 = self.w3(x_f3_all)
        x_f4 = x1 + x2

        # x_f4_all = x_f4 + x_f3 + x_f2 + x_f1 + x
        x_f4_all = x_f4 + x_f2 * self.lambda7 + x * self.lambda9
        x = F.gelu(x_f4_all)

        # x = x[..., :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = F.gelu(x)

        x = torch.reshape(x, [100, 2, 101])
        x = self.fc3(x)
        x = torch.reshape(x, [100, 1020,2])

        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)


################################################################
#  configurations
################################################################
ntrain = 1900
ntest = 100

batch_size = 100
batch_size2 = 100
learning_rate = 0.001

epochs = 8000
scheduler_step = 1000
scheduler_gamma = 0.5

modes = 16
width = 64

################################################################
# read data
################################################################

# Data is of the shape (number of samples, grid size)
data = DataSet(batch_size)
F_train, Ux_train, Uy_train, F_test, Ux_test, Uy_test, X, ux_train_mean, \
               ux_train_std, uy_train_mean, uy_train_std = data.load_data()

F_train = torch.tensor(F_train)
F_train = torch.reshape(F_train, [ntrain, 101, 1])
Ux_train = torch.tensor(Ux_train)
Uy_train = torch.tensor(Uy_train)

F_test = torch.tensor(F_test)
F_test = torch.reshape(F_test, [ntest, 101, 1])
Ux_test = torch.tensor(Ux_test)
Uy_test = torch.tensor(Uy_test)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(F_train, Ux_train, Uy_train), batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(F_test, Ux_test, Uy_test), batch_size=batch_size2,
                                          shuffle=False)

# model
model = FNO1d(modes, width).cuda()
print(count_params(model))

################################################################
# training and evaluation
################################################################
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

# learning_rate0 = 0.01
# optimizer0 = Adam([model.lambda0], lr=learning_rate0, weight_decay=0)
# scheduler0 = torch.optim.lr_scheduler.StepLR(optimizer0, step_size=scheduler_step, gamma=scheduler_gamma)
# optimizer1 = Adam([model.lambda1], lr=learning_rate0, weight_decay=0)
# scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=scheduler_step, gamma=scheduler_gamma)
# optimizer2 = Adam([model.lambda2], lr=learning_rate0, weight_decay=0)
# scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=scheduler_step, gamma=scheduler_gamma)
# optimizer3 = Adam([model.lambda3], lr=learning_rate0, weight_decay=0)
# scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=scheduler_step, gamma=scheduler_gamma)
# optimizer4 = Adam([model.lambda4], lr=learning_rate0, weight_decay=0)
# scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size=scheduler_step, gamma=scheduler_gamma)
# optimizer5 = Adam([model.lambda5], lr=learning_rate0, weight_decay=0)
# scheduler5 = torch.optim.lr_scheduler.StepLR(optimizer5, step_size=scheduler_step, gamma=scheduler_gamma)
# optimizer6 = Adam([model.lambda6], lr=learning_rate0, weight_decay=0)
# scheduler6 = torch.optim.lr_scheduler.StepLR(optimizer6, step_size=scheduler_step, gamma=scheduler_gamma)
# optimizer7 = Adam([model.lambda7], lr=learning_rate0, weight_decay=0)
# scheduler7 = torch.optim.lr_scheduler.StepLR(optimizer7, step_size=scheduler_step, gamma=scheduler_gamma)
# optimizer8 = Adam([model.lambda8], lr=learning_rate0, weight_decay=0)
# scheduler8 = torch.optim.lr_scheduler.StepLR(optimizer8, step_size=scheduler_step, gamma=scheduler_gamma)
# optimizer9 = Adam([model.lambda9], lr=learning_rate0, weight_decay=0)
# scheduler9 = torch.optim.lr_scheduler.StepLR(optimizer9, step_size=scheduler_step, gamma=scheduler_gamma)

myloss = LpLoss(size_average=False)
t0 = default_timer()
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse_x = 0
    train_mse_y = 0
    train_l2_x = 0
    train_l2_y = 0
    for x, y, z in train_loader:
        x, y, z = x.cuda(), y.cuda(), z.cuda()
        optimizer.zero_grad()
        out = model(x)
        out1 = out[:, :, 0]
        out2 = out[:, :, 1]
        msex = F.mse_loss(out1.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')
        msey = F.mse_loss(out2.view(batch_size, -1), z.view(batch_size, -1), reduction='mean')
        l2_x = myloss(out1.view(batch_size, -1), y.view(batch_size, -1))
        l2_y = myloss(out2.view(batch_size, -1), z.view(batch_size, -1))
        l2 = l2_x + l2_y
        l2.backward()  # use the l2 relative loss

        optimizer.step()
        # optimizer0.step()
        # optimizer1.step()
        # optimizer2.step()
        # optimizer3.step()
        # optimizer4.step()
        # optimizer5.step()
        # optimizer6.step()
        # optimizer7.step()
        # optimizer8.step()
        # optimizer9.step()
        train_mse_x += msex.item()
        train_mse_y += msey.item()
        train_l2_x += l2_x.item()
        train_l2_y += l2_y.item()

    scheduler.step()
    # scheduler0.step()
    # scheduler1.step()
    # scheduler2.step()
    # scheduler3.step()
    # scheduler4.step()
    # scheduler5.step()
    # scheduler6.step()
    # scheduler7.step()
    # scheduler8.step()
    # scheduler9.step()
    model.eval()
    test_l2_x = 0.0
    test_l2_y = 0.0
    with torch.no_grad():
        for x, y, z in test_loader:
            x, y, z = x.cuda(), y.cuda(), z.cuda()

            out = model(x)
            out1 = out[:, :, 0]
            out2 = out[:, :, 1]
            out1 = torch.reshape(out1, [batch_size2, 1020, 1])
            out2 = torch.reshape(out2, [batch_size2, 1020, 1])
            y = torch.reshape(y, [batch_size2, 1020, 1])
            z = torch.reshape(z, [batch_size2, 1020, 1])
            out1, out2 = data.decoder(out1, out2)
            y, z = data.decoder(y, z)
            test_l2_x += myloss(out1.view(batch_size2, -1), y.view(batch_size2, -1)).item()
            test_l2_y += myloss(out2.view(batch_size2, -1), z.view(batch_size2, -1)).item()

    train_mse_x /= len(train_loader)
    train_mse_y /= len(train_loader)
    train_l2_x /= ntrain
    train_l2_y /= ntrain
    test_l2_x /= ntest
    test_l2_y /= ntest

    t2 = default_timer()
    print(ep, t2 - t1, train_mse_x, train_mse_y, train_l2_x, train_l2_y, test_l2_x, test_l2_y)
t3 = default_timer()
print(t3-t0)
# torch.save(model.state_dict(), "D:\Transfermodels\TL5_source")