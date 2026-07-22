import torch.nn as nn
from torchvision import models
import torch


class FCNet(nn.Module):
	def __init__(self, num_classes):
		super(FCNet, self).__init__()
		self.fc1_bn = nn.BatchNorm1d(1280)
		self.fc2 = nn.Linear(1280, 32)
		# self.fc2_bn = nn.BatchNorm1d(128)
		# self.fc3 = nn.Linear(128, 32)
		self.fc4 = nn.Linear(32, num_classes)
	
	def forward(self, x):
		x = self.fc1_bn(x)
		x = self.fc2(x)
		# x = self.fc2_bn(x)
		# x = self.fc3(x)
		x = self.fc4(x)
		
		return x


model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

class conv(nn.Module):
	def __init__(self):
		super(conv, self).__init__()
		self.conv1 = model.features
		self.fc1 = nn.AdaptiveAvgPool2d((1, 1))

		self.fc1_f = nn.Flatten()
	
	def forward(self, x):
		x = self.conv1(x)
		x = self.fc1(x)
		x = self.fc1_f(x)
		return x

# Define the complete Net class with both submodules
class Net(nn.Module):
	def __init__(self, num_classes=8):
		# num_classes=8 reproduces original MANNERS-DB behaviour; OfficeDB
		# adaptation passes num_classes=9 (see OFFICEDB_MODIFICATIONS.md).
		super(Net, self).__init__()
		self.conv_module = conv()
		self.fc_module = FCNet(num_classes=num_classes)
	
	def forward(self, x):
		x = self.conv_module(x)
		x = self.fc_module(x)
		return x
