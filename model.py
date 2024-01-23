import torch 
import torch.nn as nn 

class linear_net(nn.Module): 
    def __init__(self, dropout=0.5): 
        super(linear_net, self).__init__()
        self.linear_1 = nn.Linear(784, 1200)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.linear_2  = nn.Linear(1200,1200)
        self.relu_1 = nn.ReLU()
        self.linear_3 = nn.Linear(1200,10)
        
    def forward(self, input): 
        x = self.linear_1(input)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        x = self.relu_1(x)
        x = self.dropout(x)
        x = self.linear_3(x)
        return x
    
class small_linear_net(nn.Module):
    def __init__(self, dropout=0.5): 
        super(small_linear_net, self).__init__()
        self.linear_1 = nn.Linear(784, 50)
        self.relu = nn.ReLU()
        self.linear_2  = nn.Linear(50,10)
    
    def forward(self, input): 
        x = self.linear_1(input)
        x = self.relu(x)
        x = self.linear_2(x)
        return x
    
    
