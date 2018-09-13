import torch
from torch.autograd import Variable

x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))

#heritaged by nn Module
class Model(torch.nn.Module):
    def __init__(self):
        # initial
        super(Model, self).__init__()
        # One input and One output
        self.linear = torch.nn.Linear(1, 1)
        print(self)

    def forward(self, x):
        # predict
        y_pred = self.linear(x)
        return y_pred

model = Model()

criterion = torch.nn.MSELoss(size_average=False)
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
