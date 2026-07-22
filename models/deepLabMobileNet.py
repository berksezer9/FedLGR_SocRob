import torch
# import all the necessary packages
import torch.nn as nn
import torchvision.models.segmentation as segmentation


class FCNet(nn.Module):
	def __init__(self, num_classes):
		super(FCNet, self).__init__()
		# add batchnorm
		self.fc1_bn = nn.BatchNorm1d(1344)
		self.fc2 = nn.Linear(in_features=1344, out_features=32)
		self.fc3 = nn.Linear(in_features=32, out_features=num_classes)
	
	def forward(self, x):
		x = self.fc1_bn(x)
		x = self.fc2(x)
		x = self.fc3(x)
		return x


pretrained_model = segmentation.deeplabv3_mobilenet_v3_large(pretrained=True, weights=segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT)
encoder = pretrained_model.backbone
classifier = pretrained_model.classifier


class conv(nn.Module):
	def __init__(self):
		super(conv, self).__init__()
		self.conv1 = encoder
		self.conv2 = classifier
		self.fc1 = nn.Flatten()

	
	def forward(self, x):
		x = self.conv1(x)['out']
		x = self.conv2(x)
		x = self.fc1(x)
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
