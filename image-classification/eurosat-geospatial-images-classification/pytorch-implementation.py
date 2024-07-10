# imports
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torchinfo import summary
from torchmetrics import Accuracy

# initializing transforms
TRANSFORMS = v2.Compose([
    v2.Resize((64, 64)),
    v2.Grayscale(),
    v2.ToTensor(),
])

# data loading
RAW_DATA = ImageFolder("/kaggle/input/eurosat-dataset/EuroSAT", transform=TRANSFORMS)
LOADED_DATA = DataLoader(RAW_DATA, shuffle=True, batch_size=64)


# model implementation
class ConvNet(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(3, 3), padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(0.2),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(0.2),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(0.2),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(0.2),
            nn.Flatten(),
            nn.Linear(in_features=64*16, out_features=512),
            nn.Linear(in_features=512, out_features=128),
            nn.Linear(in_features=128, out_features=32),
            nn.Linear(in_features=32, out_features=10),
        )
        
    def forward(self, x):
        return self.sequential(x)

MODEL = ConvNet().cuda()
'''print(summary(MODEL, (32, 1, 128, 128)))'''

# parameters and initializations
EPOCHS = 30
ACCURACY_CALCULATOR = Accuracy(task="multiclass", num_classes=10,).cuda()
LOSS_FUNCTION = nn.CrossEntropyLoss()
OPTIMIZER = torch.optim.Adam(MODEL.parameters())

# training loop and model saving
for epoch in range(EPOCHS):
    for iteration, (images, labels) in enumerate(LOADED_DATA):
        images, labels = images.cuda(), labels.cuda()
        predictions = MODEL(images)
        loss = LOSS_FUNCTION(predictions, labels)
        accuracy = ACCURACY_CALCULATOR(predictions, labels.float())
        OPTIMIZER.zero_grad(set_to_none=True)
        loss.backward()
        OPTIMIZER.step()
        print(f'epoch: {epoch}, iteration: {iteration}, loss: {loss:3f}, accuracy: {accuracy:3f}')

torch.save(MODEL, "eurosat.pt")
