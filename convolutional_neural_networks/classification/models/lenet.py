from torch import nn


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv_layer1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
                                         nn.ReLU(),
                                         nn.MaxPool2d(kernel_size=2, stride=2),
                                         )
        self.conv_layer2 = nn.Sequential(nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
                                         nn.ReLU(),
                                         nn.MaxPool2d(kernel_size=2, stride=2),
                                         )
        self.fc_layer1 = nn.Sequential(nn.Linear(in_features=16 * 5 * 5, out_features=120),
                                       nn.ReLU())
        self.fc_layer2 = nn.Sequential(nn.Linear(in_features=120, out_features=84),
                                       nn.ReLU())
        self.fc_layer3 = nn.Sequential(nn.Linear(in_features=84, out_features=10))

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.fc_layer1(x)
        x = self.fc_layer2(x)
        x = self.fc_layer3(x)
        return x
