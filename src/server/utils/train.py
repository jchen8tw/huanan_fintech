import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.utils.data as Data
from preprocess import get_train_data
from model import Net

device = 'cuda'
BATCH_SIZE = 20

print('--getting training data --')
print('device: ',device)
npx, npy = get_train_data()
print('size: ',npx.shape,npy.shape)

x = torch.from_numpy(npx)
y = torch.from_numpy(npy)
x = x.type(torch.float)
y = y.type(torch.float)
x = x.to(device)
y = y.to(device)

torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # random shuffle for training
    num_workers=0,              # subprocesses for loading data
)




net = Net(n_feature=99, n_hidden=50, n_output=8).to(device)

try:
    net.load_state_dict(torch.load('net.pkl'))
    print(net)
except Exception as e:
    print(e,'generate new net')
    print(net)  # net architecture

optimizer = torch.optim.Adam(net.parameters(), lr=0.0000001,betas=(0.9, 0.99))
loss_func = torch.nn.MSELoss()
for epoch in range(1000): 
    for step, (batch_x, batch_y) in enumerate(loader):  # for each training step
        try:
        # train your data...
#        batch_x = Variable(batch_x)
#        batch_y = Variable(batch_y)
            prediction = net(batch_x)
#        loss = loss_func(prediction,batch_y) #MSE
            loss = loss_func(prediction,batch_y) #crossentropy
            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients
        except KeyboardInterrupt:
            torch.save(net.to('cpu').state_dict(), 'net.pkl')
    print('Epoch: ', epoch,'| loss: ',loss.detach().cpu().numpy())
    if(epoch % 50 == 0):
        print('saved: ',epoch)
        torch.save(net.cpu().state_dict(), 'net.pkl')
        net.to(device)
        
            
    """
    print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
    batch_x.detach().numpy(), '| batch y: ', batch_y.detach().numpy(),'| loss: ',loss.detach().numpy())
    """
torch.save(net.to('cpu').state_dict(), 'net.pkl')

