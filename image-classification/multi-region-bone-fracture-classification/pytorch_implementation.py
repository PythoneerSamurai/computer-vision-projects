# imports
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torchinfo import summary

# transforms initialization
TRANSFORMS = v2.Compose([
    v2.Resize((128, 128)),
    v2.Grayscale(),
    v2.ToTensor(),
])

# data loading
RAW_DATA = ImageFolder(root="/kaggle/input/bf-dataset/bf_dataset/train", transform=TRANSFORMS)
LOADED_DATA = DataLoader(dataset=RAW_DATA, batch_size=32, shuffle=True, num_workers=4)

# model implementation
class ConvNet(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(3, 3), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(0.2),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(0.2),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(0.2),
            nn.Flatten(),
            nn.Linear(in_features=64*16*16, out_features=128),
            nn.Linear(in_features=128, out_features=16),
            nn.Linear(in_features=16, out_features=1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.sequential(x)
    
MODEL = ConvNet().cuda()
'''print(summary(MODEL, (32, 1, 128, 128)))'''

# parameters and initializations
EPOCHS = 30
LOSS_FUNCTION = nn.BCELoss()
OPTIMIZER = torch.optim.Adam(MODEL.parameters())

# training loop and model saving
for epoch in range(EPOCHS):
    for iteration, (images, labels) in enumerate(LOADED_DATA):
        images, labels = images.cuda(), labels.cuda()
        predictions = MODEL(images)
        loss = LOSS_FUNCTION(predictions, labels.view(-1, 1).float())
        OPTIMIZER.zero_grad(set_to_none=True)
        loss.backward()
        OPTIMIZER.step()
        print(f'epoch: {epoch}, iteration: {iteration}, loss: {loss:4f}')
        
torch.save(MODEL, "boneFrac.pt")
