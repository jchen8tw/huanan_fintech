import torch
import torch.nn as nn
import torch.nn.functional as F
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