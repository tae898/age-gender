import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class MLPGender(BaseModel):
    def __init__(self, num_classes=2, dropout=0.5, num_layers=None):
        super().__init__()

        self.linear1 = nn.Linear(512, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, 32)
        self.linear5 = nn.Linear(32, 16)
        self.linear6 = nn.Linear(16, 8)
        self.linear7 = nn.Linear(8, 4)
        self.linear8 = nn.Linear(4, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)

        x = F.relu(self.linear2(x))
        x = self.dropout(x)

        x = F.relu(self.linear3(x))
        x = self.dropout(x)

        x = F.relu(self.linear4(x))
        x = self.dropout(x)

        x = F.relu(self.linear5(x))
        x = self.dropout(x)

        x = F.relu(self.linear6(x))
        x = self.dropout(x)

        x = F.relu(self.linear7(x))
        x = self.dropout(x)

        x = self.linear8(x)

        return x


class MLPAge(BaseModel):
    def __init__(self, num_classes=2, dropout=0.5, num_layers=None):
        super().__init__()

        self.linear1 = nn.Linear(512, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, 32)
        self.linear5 = nn.Linear(32, 16)
        self.linear6 = nn.Linear(16, 8)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)

        x = F.relu(self.linear2(x))
        x = self.dropout(x)

        x = F.relu(self.linear3(x))
        x = self.dropout(x)

        x = F.relu(self.linear4(x))
        x = self.dropout(x)

        x = F.relu(self.linear5(x))
        x = self.dropout(x)

        x = self.linear6(x)

        return x
