import torch
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
#make input data by using numpy
a = np.zeros(500)
b = np.ones(500)

# this is input
x_t = np.arange(0,1000,1)
y_t = np.hstack((a,b))
x_t.dtype(np.float32)

#transfer to Torch
x_data = Variable(torch.from_numpy(x_t))
y_data = Variable(torch.from_numpy(y_t))

x_data = x_data.reshape(1000,1)
y_data = y_data.reshape(1000,1)

print(x_data.data.shape)

print(x_data)
#heritaged by nn Module
class Model(torch.nn.Module):
    def __init__(self):
        # initial
        super(Model, self).__init__()
        # One input and One output
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        # predict
        #  y_pred = self.linear(x)
        y_pred = F.sigmoid(self.linear(x))
        return y_pred

model = Model()

criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(500) :
    #Forward pass
    y_pred = model(x_data)

    #Compute and print loss
    loss = criterion(y_pred, y_data)

    #gradient descent
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# After training
hour_var = Variable(torch.Tensor([4.0] ))
print("\n predict ", 4,model.forward(hour_var).data[0][0])
