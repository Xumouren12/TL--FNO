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
from torch.utils.data import RandomSampler
import os

def setup_sedd(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
setup_sedd(20)

class DataSet:
    def __init__(self, bs):
        self.bs = bs
        self.F_train, self.U_train, self.F_test, self.U_test, \
        self.X, self.u_mean, self.u_std = self.load_data()


    def decoder(self, x):
        x = x * (self.u_std + 1.0e-9) + self.u_mean
        return x


    def load_data(self):
        # Source data
        # Dataset 1: Correlation length is 0.02
        # Dataset 2: Correlation length is 0.1
        # Dataset 3: Correlation length is 0.3

        file = scio.loadmat('Dataset2_TL1gauss.mat')

        s = 100
        r = 1541

        # Training target data from scratch
        # file = io.loadmat('./Data/Dataset1/Dataset_right_triangle')
        # file = io.loadmat('./Data/Dataset1/Dataset_triangle')
        # s = 100
        # r = 2295 # 1200

        f_train = file['k_train']
        u_train = file['u_train']

        f_test = file['k_test']
        u_test = file['u_test']

        f_test = np.log(f_test)
        f_train = np.log(f_train)

        xx = file['xx']
        yy = file['yy']
        xx = np.reshape(xx, (-1, 1))
        yy = np.reshape(yy, (-1, 1))
        X = np.hstack((xx, yy))

        f_train_mean = np.mean(np.reshape(f_train, (-1, s, s)), 0)
        f_train_std = np.std(np.reshape(f_train, (-1, s, s)), 0)
        f_train_mean = np.reshape(f_train_mean, (-1, s, s, 1))
        f_train_std = np.reshape(f_train_std, (-1, s, s, 1))
        F_train = np.reshape(f_train, (-1, s, s, 1))
        F_train = (F_train - f_train_mean) / (f_train_std)  # + 5.0
        F_test = np.reshape(f_test, (-1, s, s, 1))
        F_test = (F_test - f_train_mean) / (f_train_std)  # + 5.0

        u_train_mean = np.mean(np.reshape(u_train, (-1, r)), 0)
        u_train_std = np.std(np.reshape(u_train, (-1, r)), 0)
        u_train_mean = np.reshape(u_train_mean, (-1, r, 1))
        u_train_std = np.reshape(u_train_std, (-1, r, 1))
        U_train = np.reshape(u_train, (-1, r, 1))
        U_train = (U_train - u_train_mean) / (u_train_std + 1.0e-9)
        U_test = np.reshape(u_test, (-1, r, 1))
        U_test = (U_test - u_train_mean) / (u_train_std + 1.0e-9)

        return F_train, U_train, F_test, U_test, X, u_train_mean, u_train_std


    # Source
    def minibatch(self):
        # choose random indices - replace=False to avoid sampling same data
        batch_id = np.random.choice(self.F_train.shape[0], self.bs, replace=False)
        f_train = self.F_train[batch_id]
        u_train = self.U_train[batch_id]
        x_train = self.X

        Xmin = np.array([0., 0.]).reshape((-1, 2))
        Xmax = np.array([1., 1.]).reshape((-1, 2))

        return x_train, f_train, u_train, Xmin, Xmax

    def testbatch(self, num_test):
        batch_id = np.random.choice(self.F_test.shape[0], num_test, replace=False)
        f_test = self.F_test[batch_id]
        u_test = self.U_test[batch_id]
        x_test = self.X

        batch_id = np.reshape(batch_id, (-1, 1))

        return batch_id, x_test, f_test, u_test

################################################################
# fourier layer
################################################################

class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 2  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(3, self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.conv0 = SpectralConv2d_fast(self.width, self.width, 22, 22)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, 18, 18)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, 16, 16)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, 14, 14)

        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.w6 = nn.Conv2d(self.width, self.width, 1)
        self.w7 = nn.Conv2d(self.width, self.width, 1)
        self.p1 = nn.AvgPool2d(2, 2)
        self.p2 = nn.AvgPool2d(2, 2)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)
        self.fc5 = nn.Linear(625, 1541)

        dict = torch.load("D:\Transfermodels\TL1_source_1_matrix_lambda")
        self.lambda1 = dict["lambda1"]
        self.lambda3 = dict["lambda3"]
        self.lambda4 = dict["lambda4"]
        self.lambda5 = dict["lambda5"]
        self.lambda6 = dict["lambda6"]
        self.lambda9 = dict["lambda9"]

    def forward(self, x):
        x = x.to(torch.float32)
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic

        x = self.w6(x)
        x = self.p1(x)

        x = F.gelu(x)
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x_f1 = x1 + x2
        x_f1_all = x_f1

        x_f1_all = F.gelu(x_f1_all)
        x1 = self.conv1(x_f1_all)
        x2 = self.w1(x_f1_all)
        x_f2 = x1 + x2
        x_f2_all = x_f2 + x_f1 * self.lambda1

        x_f2_all = F.gelu(x_f2_all)
        x1 = self.conv2(x_f2_all)
        x2 = self.w2(x_f2_all)
        x_f3 = x1 + x2

        x_f3_all = x_f3 + x_f2 * self.lambda3 + x_f1 * self.lambda4 + x * self.lambda5
        x_f3_all = F.gelu(x_f3_all)

        x1 = self.conv3(x_f3_all)
        x2 = self.w3(x_f3_all)
        x_f4 = x1 + x2

        x_f4_all = x_f4 + x_f3 * self.lambda6 + x * self.lambda9
        x_f4_all = F.gelu(x_f4_all)

        x = self.w7(x_f4_all)
        x = self.p2(x)
        x = F.gelu(x)


        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = F.gelu(x)

        x = torch.reshape(x, [-1, 625])
        x = self.fc5(x)

        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


################################################################
# configs
################################################################

ntrain = 2000
ntest = 100

modes = 12
width = 24

batch_size = 50
batch_size2 = 50

epochs = 2500
learning_rate = 0.001
scheduler_step = 500
scheduler_gamma = 0.5

print(epochs, learning_rate, scheduler_step, scheduler_gamma)
################################################################
# load data
################################################################

data = DataSet(batch_size)
F_train, U_train, F_test, U_test, X, u_train_mean, u_train_std = data.load_data()
F_all = np.concatenate((F_train,F_test))
U_all = np.concatenate((U_train,U_test))

sampler = np.random.choice(range(2100), size=100, replace=False)
F_test = F_all[sampler, :, :, :]
F_train = np.delete(F_all, sampler, axis=0)
U_test = U_all[sampler, :, :]
U_train = np.delete(U_all, sampler, axis=0)

F_train = torch.tensor(F_train)
U_train = torch.tensor(U_train)
F_test = torch.tensor(F_test)
U_test = torch.tensor(U_test)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(F_train, U_train), batch_size=batch_size,
                                           shuffle=True, )
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(F_test, U_test), batch_size=batch_size2,
                                          shuffle=False)

################################################################
# training and evaluation
################################################################

model = FNO2d(modes, modes, width).cuda()

print(count_params(model))
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

myloss = LpLoss(size_average=False)
t0 = default_timer()
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0
    train_l2 = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        out = model(x)
        mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')
        l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
        l2.backward()  # use the l2 relative loss

        optimizer.step()
        train_mse += mse.item()
        train_l2 += l2.item()

    scheduler.step()
    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            out = model(x)
            out = torch.reshape(out, [batch_size2, 1541, 1])
            y = torch.reshape(y, [batch_size2, 1541, 1])
            out = data.decoder(out)
            y = data.decoder(y)
            test_l2 += myloss(out.view(batch_size2, -1), y.view(batch_size2, -1)).item()

    train_mse /= len(train_loader)
    train_l2 /= ntrain
    test_l2 /= ntest

    t2 = default_timer()
    print(ep, t2 - t1, train_mse, train_l2, test_l2)
t3 = default_timer()
print(t3-t0)
# torch.save(model.state_dict(), "D:\Transfermodels\TL1-3_source")