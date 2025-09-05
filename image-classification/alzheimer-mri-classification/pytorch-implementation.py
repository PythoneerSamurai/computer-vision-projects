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
])

# data loading
RAW_DATA = ImageFolder("/kaggle/input/augmented-alzheimer-mri-dataset/AugmentedAlzheimerDataset", transform=TRANSFORMS)
LOADED_DATA = DataLoader(RAW_DATA, shuffle=True, batch_size=32)

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
            nn.Linear(in_features=64**2, out_features=512),
            nn.Linear(in_features=512, out_features=128),
            nn.Linear(in_features=128, out_features=32),
            nn.Linear(in_features=32, out_features=4),
        )
        
    def forward(self, x):
        return self.sequential(x)

MODEL = ConvNet().cuda()
'''print(summary(MODEL, (32, 1, 128, 128)))'''

# parameters and initializations
EPOCHS = 30
ACCURACY_CALCULATOR = Accuracy(task="multiclass", num_classes=4,).cuda()
LOSS_FUNCTION = nn.CrossEntropyLoss()
OPTIMIZER = torch.optim.Adam(MODEL.parameters())

# for early stopping
accuracyList = []
lossList = []
highestAccuracy = float
lowAccuracyEpochList = []

# early stopping
def early_stopping(highestAccuracyEpoch, lowAccuracyEpochList):
    stopEarly = False
    if len(lowAccuracyEpochList) == 5:
        for num in range(5):
            if lowAccuracyEpochList[num] == (highestAccuracyEpoch + num + 1):
                stopEarly = True
            else:
                stopEarly = False
        return stopEarly

# training loop and model saving
for epoch in range(EPOCHS):
    for iteration, (images, labels) in enumerate(LOADED_DATA):
        images, labels = images.cuda(), labels.cuda()
        predictions = MODEL(images)
        loss = LOSS_FUNCTION(predictions, labels)
        lossList.append(loss.cpu().float())
        accuracy = ACCURACY_CALCULATOR(predictions, labels.float())
        accuracyList.append(accuracy.cpu().float())
        OPTIMIZER.zero_grad(set_to_none=True)
        loss.backward()
        OPTIMIZER.step()

  # average accuracy and average loss per epoch
    acc = (sum(accuracyList)/len(accuracyList)).cpu().float()
    loss = (sum(lossList)/len(lossList)).cpu().float()
    print(f'epoch: {epoch}, iteration: {iteration}, loss: {loss:3f}, accuracy: {acc:3f}')

  # code for early stopping, criterion = no accuracy increase for 5 epochs
    if epoch == 0 or (acc > highestAccuracy and acc != float(1)):
        highestAccuracy = acc
        highestAccuracyEpoch = epoch
        lowAccuracyEpochList = []
        if highestAccuracy != float(1):
            torch.save(MODEL, "alzheimer.pt")  # model saved everytime a higher accuracy is attained, however float(1) is discarded, as the model is probably overfitted at that accuracy
    elif acc <= highestAccuracy or acc == float(1):
        lowAccuracyEpochList.append(epoch)
        
    if lowAccuracyEpochList is not None:
        shouldStop = early_stopping(highestAccuracyEpoch, lowAccuracyEpochList)
        if shouldStop is True:
            break
        else:
            pass

