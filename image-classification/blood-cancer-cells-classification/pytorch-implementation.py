'''
Note:
The labels are as follows:
1. Benign
2. [Malignant] Pre-B
3. [Malignant] Pro-B
4. [Malignant] early Pre-B
'''

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
    v2.Resize((128, 128)),
    v2.Grayscale(),
    v2.ToTensor(),
    v2.ToDtype(torch.float32),
])

# data loading
RAW_DATA = ImageFolder("/kaggle/input/blood-cell-cancer-all-4class/Blood cell Cancer [ALL]", transform=TRANSFORMS)
LOADED_DATA = DataLoader(RAW_DATA, shuffle=True, batch_size=64)

# model implementation
class ConvNet(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=(3, 3), padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(0.2),
            nn.BatchNorm2d(num_features=128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(0.2),
            nn.BatchNorm2d(num_features=128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(0.2),
            nn.BatchNorm2d(num_features=128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(0.2),
            nn.Flatten(),
            nn.Linear(in_features=128*64, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=4),
        )
        
    def forward(self, x):
        return self.sequential(x)

MODEL = ConvNet().cuda()
'''print(summary(MODEL, (32, 1, 128, 128)))'''

# parameters and initializations
EPOCHS = 20
ACCURACY_CALCULATOR = Accuracy(task="multiclass", num_classes=4).cuda()
LOSS_FUNCTION = nn.CrossEntropyLoss()
OPTIMIZER = torch.optim.Adam(MODEL.parameters())

# training loop and model saving
for epoch in range(EPOCHS):
    for iteration, (images, labels) in enumerate(LOADED_DATA):
        images, labels = images.cuda(), labels.cuda()
        predictions = MODEL(images)
        loss = LOSS_FUNCTION(predictions, labels)
        accuracy = ACCURACY_CALCULATOR(predictions, labels)
        OPTIMIZER.zero_grad(set_to_none=True)
        loss.backward()
        OPTIMIZER.step()
        if iteration % 20 == 0:
            print(f'epoch: {epoch}, iteration: {iteration}, loss: {loss:3f}, accuracy: {accuracy:3f}')

torch.save(MODEL, "bloodCancer.pt")
