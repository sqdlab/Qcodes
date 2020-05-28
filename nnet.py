import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np
import uqtools as uq
import pandas as pd


# class CNN(nn.Module):
#     """
#     2xCNN, 1xpooling, 1xlinear
#     """

#     def __init__(self, n_time_steps, k_features):
#         super(CNN, self).__init__()
#         conv1_out = 16 # out channels
#         conv1_kernel = 10
#         conv2_out = 32
#         conv2_kernel = 5
#         pool_ks = 3
#         lin1_in = (n_time_steps-(conv2_kernel-1)-(conv1_kernel-1))//pool_ks * conv2_out
#         lin1_out = lin1_in//2
#         output_size = 3

#         self.conv1 = nn.Conv1d(k_features, conv1_out, kernel_size=conv1_kernel),
#         self.relu  = nn.ReLU(),
#         self.conv2  = nn.Conv1d(conv1_out, conv2_out, kernel_size=conv2_kernel),
#         self.pool  = nn.MaxPool1d(pool_ks),
#         self.flatten  = nn.Flatten(),
#         self.dropout  = nn.Dropout(),
#         self.lin1  = nn.Linear(lin1_in, lin1_out),
#         self.lin2  = nn.Linear(lin1_out, output_size)
    
#     def forward(self, X):
#         X = self.conv1(X)
#         X = self.relu(X)
#         X = self.conv2(X)
#         X = self.relu(X)
#         X = self.pool(X)
#         X = self.flatten(X)
#         X = self.dropout(X)
#         X = self.lin1(X)
#         X = self.relu(X)
#         X = self.lin2(X)

#         return X
# net = CNN(n_time_steps, k_features)

def adjust_m4i_data(source_data):
    """ 
    Input:  (iteration, segment, sample, channel)
    """
    print("Begin reshaping ... {}".format(source_data.shape))
    # Remove channel (last index) and cute data shape to fit into model
    # model currenly has a window of 128, need to make adjustable
    # taking first ten just for testing
    # if(source_data..shape[-1] ! = 1):
    #     log.warning("Unexpected shape in channels")
    new_data = source_data[0:,:,0:128,0]
    # collapse second dimention
    new_data = new_data.reshape(new_data.shape[0], new_data.shape[2])
    # unpack complex numbers [I0+iQ0, I1+iQ1, ...] into [I0, Q0, I1, Q1, ...] 
    # Dubious to convert to df just for this ,  also could do for in range instead of adding dim
    # np.array([[np.real(j),np.imag(j)] for i, j in new_data])  
    converted_data = uq.pandas.unpack_complex(pd.DataFrame(new_data.copy()))  
    # add in column dimention for model requirement
    data_shape = converted_data.shape
    converted_data = converted_data.values.reshape(data_shape[0], -1, data_shape[1])
    # model input must be a tensor
    print("End reshaping ...{}".format(converted_data.shape))
    return torch.tensor(converted_data)


""" Settings for the model, these just agree with those in the pth file """
n_time_steps = 256 #X_train.shape[2] 
k_features = 1 #X_train.shape[1]

model_type = "C"
conv1_out = 16 # out channels
conv1_kernel = 10
conv2_out = 32
conv2_kernel = 5
pool_ks = 3
lin1_in = (n_time_steps-(conv2_kernel-1)-(conv1_kernel-1))//pool_ks * conv2_out
lin1_out = lin1_in//2
output_size = 3

net = nn.Sequential(
    nn.Conv1d(k_features, conv1_out, kernel_size=conv1_kernel),
    nn.ReLU(),
    nn.Conv1d(conv1_out, conv2_out, kernel_size=conv2_kernel),
    nn.ReLU(),
    nn.MaxPool1d(pool_ks),
    nn.Flatten(),
    nn.Dropout(),
    nn.Linear(lin1_in, lin1_out),
    nn.ReLU(),
    nn.Linear(lin1_out, output_size)
)

# model trained on other data from summer
saved_model = 'C:/Users/Experiment/Documents/Qcodes/models/cnnC_f_1590133074_batch.pth'
net.load_state_dict(torch.load(saved_model))
net.eval()
# Gpu?
net.double()