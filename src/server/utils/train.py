import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.utils.data as Data
import torch.nn.functional as F
from preprocess import get_train_data

device = 'cuda'
BATCH_SIZE = 20000

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

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden,n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        #nn.init.xavier_uniform_(self.hidden, gain=nn.init.calculate_gain('relu'))
        self.hidden2 = torch.nn.Linear(n_hidden,n_hidden)
        self.hidden3 = torch.nn.Linear(n_hidden,n_hidden)
        self.hidden4 = torch.nn.Linear(n_hidden,n_hidden)
        self.hidden5 = torch.nn.Linear(n_hidden,n_hidden)
        self.hidden6 = torch.nn.Linear(n_hidden,n_hidden)
        self.hidden7 = torch.nn.Linear(n_hidden,n_hidden)
        self.hidden8 = torch.nn.Linear(n_hidden,n_hidden)
        self.hidden9 = torch.nn.Linear(n_hidden,n_hidden)
        self.hidden10 = torch.nn.Linear(n_hidden,n_hidden)
        self.hidden11 = torch.nn.Linear(n_hidden,n_hidden)
        self.hidden12 = torch.nn.Linear(n_hidden,n_hidden)
        self.hidden13 = torch.nn.Linear(n_hidden,n_hidden)
        self.hidden14 = torch.nn.Linear(n_hidden,n_hidden)
        self.hidden15 = torch.nn.Linear(n_hidden,n_hidden)
        self.hidden16 = torch.nn.Linear(n_hidden,n_hidden)
        self.hidden17 = torch.nn.Linear(n_hidden,n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.hidden4(x))
        x = F.relu(self.hidden5(x))
        x = F.relu(self.hidden6(x))
        x = F.relu(self.hidden7(x))
        x = F.relu(self.hidden8(x))
        x = F.relu(self.hidden9(x))
        x = F.relu(self.hidden10(x))
        x = F.relu(self.hidden11(x))
        x = F.relu(self.hidden12(x))
        x = F.relu(self.hidden13(x))
        x = F.relu(self.hidden14(x))
        x = F.relu(self.hidden15(x))
        x = F.relu(self.hidden16(x))
        x = F.relu(self.hidden17(x))
        x = self.predict(x)             # linear output
        return x


net = Net(n_feature=99, n_hidden=50, n_output=8).to(device)

try:
    net.load_state_dict(torch.load('net_norm.pkl'))
    print(net)
except Exception as e:
    print(e,'generate new net')
    print(net)  # net architecture

optimizer = torch.optim.Adam(net.parameters(), lr=0.00000001,betas=(0.9, 0.99))
loss_func = torch.nn.MSELoss()
loss_func_class = torch.nn.CrossEntropyLoss()
#Todo
#use cross entropy for case classification output
#use MSE for otherwise
for epoch in range(1000): 
    for step, (batch_x, batch_y) in enumerate(loader):  # for each training step
        try:
        # train your data...
#        batch_x = Variable(batch_x)
#        batch_y = Variable(batch_y)
            prediction = net(batch_x)
#            print(batch_x,batch_y)
#        loss = loss_func(prediction,batch_y) #MSE
            loss_mse = loss_func(prediction[:,0:4],batch_y[:,0:4]) 
            loss_cross = loss_func_class(prediction[:,4:],torch.argmax(batch_y[:,4:],dim=1))#crossentropy
            optimizer.zero_grad()   # clear gradients for next train
            loss_mse.backward(retain_graph=True)         # backpropagation, compute gradients
            loss_cross.backward()
            optimizer.step()        # apply gradients
        except KeyboardInterrupt:
            torch.save(net.to('cpu').state_dict(), 'net_norm.pkl')
    print('Epoch: ', epoch,'| loss: ',loss_mse.detach().cpu().numpy() + loss_cross.detach().cpu().numpy())
    if(epoch % 50 == 0):
        print('saved: ',epoch)
        torch.save(net.cpu().state_dict(), 'net_norm.pkl')
        net.to(device)
        
            
    """
    print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
    batch_x.detach().numpy(), '| batch y: ', batch_y.detach().numpy(),'| loss: ',loss.detach().numpy())
    """
torch.save(net.to('cpu').state_dict(), 'net_norm.pkl')

