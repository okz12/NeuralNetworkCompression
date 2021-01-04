import torch.nn as nn
import torch.nn.functional as F

class teacherNet(nn.Module):
    
    def __init__(self):
        super(teacherNet, self).__init__()
        
        self.name = 'teacherNet'
        
        self.fc1 = nn.Linear(28*28, 1200)
        self.fc2 = nn.Linear(1200, 300)
        self.fc3 = nn.Linear(300, 100)
        self.fc4 = nn.Linear(100, 10)
        
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(F.relu(x))
        x = self.fc2(F.relu(x))
        x = self.fc3(F.relu(x))
        x = self.fc4(F.relu(x))
        return x


class studentNet(nn.Module):
    
    def __init__(self):
        super(studentNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 200)
        self.fc2 = nn.Linear(200, 10)
        
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(F.relu(x))
        x = self.fc2(F.relu(x))
        return x