import torch.nn as nn
import torch
# from functions import ReverseLayerF


outputSize=6

class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.IN1 = nn.InstanceNorm2d(64, affine=True)
        self.relu1 = nn.ReLU(True)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(192)
        self.IN2 = nn.InstanceNorm2d(192, affine=True)
        self.relu2 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(384)
        self.IN3 = nn.InstanceNorm2d(384, affine=True)
        self.relu3 = nn.ReLU(True)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.IN4 = nn.InstanceNorm2d(256, affine=True)
        self.relu4 = nn.ReLU(True)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.bn5 = nn.BatchNorm2d(256)
        self.IN5 = nn.InstanceNorm2d(256, affine=True)
        self.relu5 = nn.ReLU(True)

    def forward(self, input_data, alpha):
        input_data = input_data.expand(input_data.data.shape[0], 3, 17, 17)
        x = self.conv1(input_data)
        x = self.bn1(x)
        x = self.IN1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.IN2(x)
        x = self.relu2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.IN3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.IN4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.IN5(x)
        x = self.relu5(x)
        feature = x.view(-1, 256 * outputSize * outputSize)
        return feature


class Classifier(nn.Module):
    def __init__(self, output_num):
        super(Classifier, self).__init__()
        self.drop1 = nn.Dropout2d()
        self.fc1 = nn.Linear(256 * outputSize * outputSize, 4096)
        self.bn1 = nn.BatchNorm1d(4096)
        self.relu1 = nn.ReLU(True)
        self.drop2 = nn.Dropout2d()
        self.fc2 = nn.Linear(4096, 4096)
        self.bn2 = nn.BatchNorm1d(4096)
        self.relu2 = nn.ReLU(True)
        self.fc3 = nn.Linear(4096, output_num)
        self.output = output_num

    def forward(self, x):
        x = self.drop1(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.drop2(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


class Adver(nn.Module):
    def __init__(self):
        super(Adver, self).__init__()
        self.ad_layer1 = nn.Linear(256 * outputSize * outputSize, 1024)
        self.ad_layer2 = nn.Linear(1024, 1024)
        self.ad_layer3 = nn.Linear(1024, 1)
        self.ad_layer1.weight.data.normal_(0, 0.01)
        self.ad_layer2.weight.data.normal_(0, 0.01)
        self.ad_layer3.weight.data.normal_(0, 0.3)
        self.ad_layer1.bias.data.fill_(0.0)
        self.ad_layer2.bias.data.fill_(0.0)
        self.ad_layer3.bias.data.fill_(0.0)
        self.relu1 = nn.LeakyReLU()
        self.relu2 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.ad_layer3(x)
        x = self.sigmoid(x)
        return x
