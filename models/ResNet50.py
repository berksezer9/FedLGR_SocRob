import torch.nn as nn
from torchvision import models
import torch


class FCNet(nn.Module):
	def __init__(self, num_classes):
		super(FCNet, self).__init__()
		# 2048 = ResNet-50's pooled feature width (vs. MobileNetV2's 1280) --
		# see OFFICEDB_MODIFICATIONS.md item 37.
		self.fc1_bn = nn.BatchNorm1d(2048)
		self.fc2 = nn.Linear(2048, 32)
		self.fc4 = nn.Linear(32, num_classes)

	def forward(self, x):
		x = self.fc1_bn(x)
		x = self.fc2(x)
		x = self.fc4(x)

		return x


model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.fc = nn.Identity()


class conv(nn.Module):
	def __init__(self):
		super(conv, self).__init__()
		# torchvision's ResNet.forward() already does avgpool + flatten before
		# self.fc, so with fc replaced by Identity, calling the whole model
		# returns the flattened 2048-dim pooled feature directly -- no need
		# for MobileNet.py's separate AdaptiveAvgPool2d/Flatten bolted on.
		self.backbone = model

	def forward(self, x):
		x = self.backbone(x)
		return x


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
