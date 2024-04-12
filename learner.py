import torch 
import torch.nn as nn  
import torch.nn.functional as F  


class Learner(nn.Module):
    def __init__(self, input_dim=2048, drop_p=0.0):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(), 
            nn.Dropout(0.6),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.drop_p = 0.6 
        self.weight_init()
        self.vars = nn.ParameterList() 
        
        
        for i, param in enumerate(self.classifier.parameters()):
            self.vars.append(param) 
        
        
    def weight_init(self):
        for layer in self.classifier:
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight)
    
# 이러한 훈련방식을 쓰면 직접 layer의 parameter를 좀 더 세부적으로 조절가능
    def forward(self, x, vars=None):
        if vars is None:
            vars = self.vars 
        
        x = F.linear(x, vars[0], vars[1])
        x = F.relu(x)
        x = F.dropout(x, self.drop_p, training=self.training)
        x = F.linear(x, vars[2], vars[3])
        x = F.dropout(x, self.drop_p, training=self.training)
        x = F.linear(x, vars[4], vars[5]) 
        return torch.sigmoid(x) 
    

    def parameters(self):
        return self.vars 