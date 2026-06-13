import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
n_samples = 1000
sample_dim = 2
X = np.random.randn(n_samples, sample_dim)
w = np.array([2, -3.4]).reshape(2,)
b = 4.2
e = np.random.randn(n_samples, ) * 0.01
y = X @ w + b + e
print(y.shape)

# numpy -> tensor

X = torch.from_numpy(X).float()
y = torch.from_numpy(y).float()

from torch.utils.data  import Dataset, DataLoader

class MyData(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, index: int):
        return self.X[index], self.y[index]
    
mydata = MyData(X,y)
batch_size = 32
train_dataloader = DataLoader(mydata, batch_size=batch_size, shuffle=True, drop_last=True)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(2))
        self.b = nn.Parameter(torch.ones(1,))
        
    def forward(self,x):
        return x @ self.w + self.b
    
net = Net()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3) 
epochs = 200
criterion = nn.MSELoss()
loss_all = []
for epoch in range(epochs):
    loss_epoch = 0
    for i, (data_batch, label_batch) in enumerate(train_dataloader):
        output = net(data_batch)
        optimizer.zero_grad()
        loss = criterion(output,label_batch)
        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()
    print(f"loss={loss.item()}")
    loss_all.append(loss.item()/len(train_dataloader))
print(w,b) 
for param in net.parameters():
    print(param)    

plt.figure()
plt.plot(loss_all,'b-')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
