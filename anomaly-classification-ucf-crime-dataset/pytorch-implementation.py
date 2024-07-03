# imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torchinfo import summary

# parameters
BATCH_SIZE = 32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 30

# data loading and transformations
TRANSFORMS = v2.Compose([
    v2.Grayscale(),
    v2.ToTensor(),
])

TRAIN_DATA = ImageFolder("/kaggle/input/ucf-dataset/Train", transform=TRANSFORMS)
LOADED_DATA = DataLoader(TRAIN_DATA, BATCH_SIZE, pin_memory=True)

# model implementation
class ConvNet(nn.Module):
    def __init__(self, input_channels=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(num_features=input_channels)
        self.mp = nn.MaxPool2d(kernel_size=(2, 2))
        self.c1 = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=(3, 3), padding='same')
        self.dp = nn.Dropout2d(0.2)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.c2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding='same')
        self.l1 = nn.Linear(in_features=16384, out_features=128)
        self.l2 = nn.Linear(in_features=128, out_features=16)
        self.l3 = nn.Linear(in_features=16, out_features=1)
        
    def forward(self, x):
        bn1 = self.bn1(x)
        c1 = self.c1(bn1)
        dp = self.dp(F.relu(c1))
        bn2 = self.bn2(dp)
        c2 = self.c2(self.mp(bn2))
        dp = self.dp(F.relu(c2))
        dp = dp.view(-1, 16384)
        l1 = self.l1(dp)
        l2 = self.l2(l1)
        l3 = self.l3(l2)
        
        return F.sigmoid(l3)
    
MODEL = ConvNet()
MODEL.to(DEVICE)
print(summary(MODEL, (1, 1, 64, 64)))

# loss function and optimizer object initialization
LOSS_FUNCTION = nn.BCELoss()
OPTIMIZER = torch.optim.Adam(model.parameters())

# training loop
for epoch in range(EPOCHS):
    for images, labels in LOADED_DATA:
        images.to(DEVICE)
        predictions = MODEL(images)
        labels.view(-1, 1)
        labels.to(DEVICE)
        loss = LOSS_FUNCTION(predictions, labels)
        OPTIMIZER.zero_grad(set_to_none=True)
        loss.backward()
        OPTIMIZER.step()
        print(f'epoch:{epoch}, loss:{loss}')
