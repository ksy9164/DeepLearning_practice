import os                       # import os for save result
import torch                    # import torch for using DataLoader and Optimizer
import torchvision              # import torchvision for Dataset
import torch.nn as nn           # import nn to make Neaural_Network module
from torchvision import transforms
from torchvision.utils import save_image # for save real,fake image 
import torch.nn.functional as F

# Hyper-parameters
latent_size = 64    # G input noise size
hidden_size = 256   # G hidden layer size
image_size = 784    # Mnist image size
num_epochs = 200    # epoch
batch_size = 100    # batch
sample_dir = 'samples'  # Directory for saving result

# Make "transform" function for Converting Data to Tensor and make it Normalized
transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,0.5,0.5),
                                     std=(0.5,0.5,0.5))])

# Select Mnist Data
mnist = torchvision.datasets.MNIST(root='./../',
                                  train=True,
                                  transform=transform,
                                  download=True)

# Data Loader
data_loader = torch.utils.data.DataLoader(dataset=mnist,
                                          batch_size=batch_size,
                                          shuffle=True)

# Discriminator Model
class D(nn.Module):

    def __init__(self) :
        super(D,self).__init__()
        # define function for forwarding
        self.layer1 = nn.Linear(image_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, 1)
        self.act = nn.LeakyReLU(0.2)
        self.act_sig = nn.Sigmoid()

    def forward(self,x) :
        x = self.act(self.layer1(x))
        x = self.act(self.layer2(x))
        return self.act_sig(self.layer3(x))

# Generator Model
class G(nn.Module):

    def __init__(self) :
        super(G,self).__init__()
        self.layer1 = nn.Linear(latent_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, image_size)
        self.act = nn.ReLU()
        self.act_tan = nn.Tanh()

    def forward(self,x) :
        x = self.act(self.layer1(x))
        x = self.act(self.layer2(x))
        return self.act_tan(self.layer3(x))

G = G()
D = D()

# Optimizer is Binary Cross Entropy
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)

# ??? 
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

# reset both grades
def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()

total_step = len(data_loader)

# start training!!
tatal_step = len(data_loader)

for epoch in range(num_epochs): # while ( 0 ~ epochs)
    for i,(images,_) in enumerate(data_loader): # i is index & image is data ( i = 0 ~ 600 )
        images = images.reshape(batch_size, -1) # in one for loop, it has 1 batch images ( 100 )

        # Create Label for Discriminator
        real_labels = torch.ones(batch_size, 1) # for real data
        fake_labels = torch.zeros(batch_size, 1) # for fake data

        # ================ For Discriminator =================== #
        output = D(images)  # FeedForward in Discriminator
        d_loss_real = criterion(output, real_labels) # The loss of Discriminator
        real_score = output # rate of accuracy

        # Make Noise input for Generator
        z = torch.randn(batch_size, latent_size)
        fake_image = G(z)       # Make Image by using Generator ( Fake Images )
        output = D(fake_image)  # Discriminator distinguish the Fake Images
        d_loss_fake = criterion(output, fake_labels) # Calculate Loss
        fake_score = output     # rate of accuracy

        # Combine loss
        d_loss = d_loss_real + d_loss_fake

        reset_grad()        # init grad
        d_loss.backward()   # calculate the grad
        d_optimizer.step()  # update prameters

        
        # ================== For Generator ===================== #

        # Make Noise input for Generator
        z = torch.randn(batch_size, latent_size)
        fake_image = G(z)           # Generator make image
        output = D(fake_image)      # Tested by Discriminator

        g_loss = criterion(output, real_labels)

        reset_grad()        # init grad
        g_loss.backward()   # calculate the grad
        g_optimizer.step()
        
        # Print Process
        if (i+1) % 200 == 0:
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}' 
                  .format(epoch, num_epochs, i+1, total_step, d_loss.item(), g_loss.item(), 
                          real_score.mean().item(), fake_score.mean().item()))
    
    # Save real images
    if (epoch+1) == 1:
        images = images.reshape(images.size(0), 1, 28, 28)
        save_image(denorm(images), os.path.join(sample_dir, 'real_images.png'))
    
    # Save sampled images
    fake_image = fake_image.reshape(fake_image.size(0), 1, 28, 28)
    save_image(denorm(fake_image), os.path.join(sample_dir, 'fake_image-{}.png'.format(epoch+1)))

# Save the model checkpoints 
torch.save(G.state_dict(), 'G.ckpt')
torch.save(D.state_dict(), 'D.ckpt')
