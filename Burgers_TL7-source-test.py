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

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)
        self.fc3 = nn.Linear(128,256)

        self.lambda0 = nn.Parameter(torch.rand(128))
        self.lambda1 = nn.Parameter(torch.rand(128))
        self.lambda2 = nn.Parameter(torch.rand(128))
        self.lambda3 = nn.Parameter(torch.rand(128))
        self.lambda4 = nn.Parameter(torch.rand(128))
        self.lambda5 = nn.Parameter(torch.rand(128))
        self.lambda6 = nn.Parameter(torch.rand(128))
        self.lambda7 = nn.Parameter(torch.rand(128))
        self.lambda8 = nn.Parameter(torch.rand(128))
        self.lambda9 = nn.Parameter(torch.rand(128))


    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        # x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x_f1 = x1 + x2
        x_f1_all = x_f1 + self.lambda0 * x
        x_f1_all = F.gelu(x_f1_all)

        x1 = self.conv1(x_f1_all)
        x2 = self.w1(x_f1_all)
        x_f2 = x1 + x2
        x_f2_all = x_f2 + self.lambda1 * x_f1 + self.lambda2 * x
        x_f2_all = F.gelu(x_f2_all)

        x1 = self.conv2(x_f2_all)
        x2 = self.w2(x_f2_all)
        x_f3 = x1 + x2
        x_f3_all = x_f3 + self.lambda3 * x_f2 + self.lambda4 * x_f1 + self.lambda5 * x
        x_f3_all = F.gelu(x_f3_all)

        x1 = self.conv3(x_f3_all)
        x2 = self.w3(x_f3_all)
        x_f4 = x1 + x2
        x_f4_all = x_f4 + self.lambda6 * x_f3 + self.lambda7 * x_f2 + self.lambda8 * x_f1 + self.lambda9 * x
        x = F.gelu(x_f4_all)


        # x = x[..., :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = F.gelu(x)
        x = torch.reshape(x,[-1,128])
        x = self.fc3(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)


################################################################
#  configurations
################################################################
ntrain = 500
ntest = 200

sub = 1  # subsampling rate
h = 128 # total grid size divided by the subsampling rate
s = 256

batch_size = 10
batch_size2 = 100
learning_rate = 0.001

epochs = 1500
step_size = 150
gamma = 0.5

modes = 16
width = 64

################################################################
# read data
################################################################

# Data is of the shape (number of samples, grid size)
data__1 = scio.loadmat('burgers_TL7_source.mat')
data__2 = scio.loadmat('burgers_TL7_source_u.mat')
x_train = data__1['input'][:ntrain, :]
y_train = data__2['output'][:ntrain, 99, :]

x_test = data__1['input'][2000:,:]
y_test = data__2['output'][2000:, 99, :]

# f_train_mean = np.mean(np.reshape(x_train, (-1, h)), 0)
# f_train_std = np.std(np.reshape(x_train, (-1, h)), 0)
# f_train_mean = np.reshape(f_train_mean, (-1, h, 1))
# f_train_std = np.reshape(f_train_std, (-1, h, 1))
# F_train = np.reshape(x_train, (-1, h, 1))
# F_train = (F_train - f_train_mean) / (f_train_std)  # + 5.0
# F_test = np.reshape(x_test, (-1, h, 1))
# F_test = (F_test - f_train_mean) / (f_train_std)  # + 5.0
#
# u_train_mean = np.mean(np.reshape(y_train, (-1, s)), 0)
# u_train_std = np.std(np.reshape(y_train, (-1, s)), 0)
# u_train_mean = np.reshape(u_train_mean, (-1, s, 1))
# u_train_std = np.reshape(u_train_std, (-1, s, 1))
# U_train = np.reshape(y_train, (-1, s, 1))
# U_train = (U_train - u_train_mean) / (u_train_std)
# U_test = np.reshape(y_test, (-1, s, 1))
# U_test = (U_test - u_train_mean) / (u_train_std)
#
# def decoder(x):
#     x = x * (u_train_std) + u_train_mean
#     return x
F_train = x_train.reshape(ntrain, h, 1)
F_test = x_test.reshape(ntest, h, 1)
U_train = y_train.reshape(ntrain,s,1)
U_test = y_test.reshape(ntest,s,1)


x_train = torch.from_numpy(F_train)
x_train = torch.tensor(x_train,dtype=torch.float32)
x_test = torch.from_numpy(F_test)
x_test = torch.tensor(x_test,dtype=torch.float32)
y_train = torch.from_numpy(U_train)
y_train = torch.tensor(y_train,dtype=torch.float32)
y_test = torch.from_numpy(U_test)
y_test = torch.tensor(y_test,dtype=torch.float32)


train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size2,
                                          shuffle=False)

# model
model = FNO1d(modes, width).cuda()
print(count_params(model))

################################################################
# training and evaluation
################################################################
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

learning_rate0 = 0.01
optimizer0 = torch.optim.Adam([model.lambda0], lr=learning_rate0, weight_decay=0)
scheduler0 = torch.optim.lr_scheduler.StepLR(optimizer0, step_size=step_size, gamma=gamma)
optimizer1 = torch.optim.Adam([model.lambda1], lr=learning_rate0, weight_decay=0)
scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=step_size, gamma=gamma)
optimizer2 = torch.optim.Adam([model.lambda2], lr=learning_rate0, weight_decay=0)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=step_size, gamma=gamma)
optimizer3 = torch.optim.Adam([model.lambda3], lr=learning_rate0, weight_decay=0)
scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=step_size, gamma=gamma)
optimizer4 = torch.optim.Adam([model.lambda4], lr=learning_rate0, weight_decay=0)
scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size=step_size, gamma=gamma)
optimizer5 = torch.optim.Adam([model.lambda5], lr=learning_rate0, weight_decay=0)
scheduler5 = torch.optim.lr_scheduler.StepLR(optimizer5, step_size=step_size, gamma=gamma)
optimizer6 = torch.optim.Adam([model.lambda6], lr=learning_rate0, weight_decay=0)
scheduler6 = torch.optim.lr_scheduler.StepLR(optimizer6, step_size=step_size, gamma=gamma)
optimizer7 = torch.optim.Adam([model.lambda7], lr=learning_rate0, weight_decay=0)
scheduler7 = torch.optim.lr_scheduler.StepLR(optimizer7, step_size=step_size, gamma=gamma)
optimizer8 = torch.optim.Adam([model.lambda8], lr=learning_rate0, weight_decay=0)
scheduler8 = torch.optim.lr_scheduler.StepLR(optimizer8, step_size=step_size, gamma=gamma)
optimizer9 = torch.optim.Adam([model.lambda9], lr=learning_rate0, weight_decay=0)
scheduler9 = torch.optim.lr_scheduler.StepLR(optimizer9, step_size=step_size, gamma=gamma)

t0 = default_timer()
result_list = list()
myloss = LpLoss(size_average=False)
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
        optimizer0.step()
        optimizer1.step()
        optimizer2.step()
        optimizer3.step()
        optimizer4.step()
        optimizer5.step()
        optimizer6.step()
        optimizer7.step()
        optimizer8.step()
        optimizer9.step()
        train_mse += mse.item()
        train_l2 += l2.item()

    scheduler.step()
    scheduler0.step()
    scheduler1.step()
    scheduler2.step()
    scheduler3.step()
    scheduler4.step()
    scheduler5.step()
    scheduler6.step()
    scheduler7.step()
    scheduler8.step()
    scheduler9.step()
    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()

            out = model(x)
            # out = torch.reshape(out, [batch_size2, s, 1])
            # y = torch.reshape(y, [batch_size2, s, 1])
            # out = decoder(out)
            # y = decoder(y)
            test_l2 += myloss(out.view(batch_size2, -1), y.view(batch_size2, -1)).item()
            if ep == 1499:
                result_list.append(out.reshape(batch_size2, -1))
                result_list.append(y.reshape(batch_size2, -1))

    train_mse /= len(train_loader)
    train_l2 /= ntrain
    test_l2 /= ntest

    t2 = default_timer()
    print(ep, t2 - t1, train_mse, train_l2, test_l2)

t3 = default_timer()
print(t3-t0)
torch.save(model.state_dict(), "D:\Transfermodels\TL7_Burgers_source4")
# utest1 = np.reshape(result_list[0], [-1,7840])
# utest_2 = np.reshape(result_list[1], [-1,7840])
print(len(result_list))
utest3 = result_list[0]
print(utest3.shape)
utest_4 = result_list[1]
print(utest_4.shape)
# plt.plot(utest1[0,:])
# plt.plot(utest_2[0,:])
plt.plot(utest_4[2,:], )
plt.plot(utest3[2,:],color = 'red')
plt.show()
plt.plot(utest_4[0,:], )
plt.plot(utest3[0,:],color = 'red')
plt.show()



