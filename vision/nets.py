import torch
import torch.nn as nn
import torch.nn.functional as F

"""found in:
https://github.com/pytorch/examples/blob/main/mnist/main.py
"""

class MNISTCNN(nn.Module):
	def __init__(self):
		super(MNISTCNN, self).__init__()
		self.conv1 = nn.Conv2d(1, 32, 3, 1)
		self.conv2 = nn.Conv2d(32, 64, 3, 1)
		self.dropout1 = nn.Dropout(0.25)
		self.dropout2 = nn.Dropout(0.5)
		self.fc1 = nn.Linear(9216, 128)
		self.fc2 = nn.Linear(128, 10)

	def forward(self, x):
		x = self.conv1(x)
		x = F.relu(x)
		x = self.conv2(x)
		x = F.relu(x)
		x = F.max_pool2d(x, 2)
		x = self.dropout1(x)
		x = torch.flatten(x, 1)
		x = self.fc1(x)
		x = F.relu(x)
		x = self.dropout2(x)
		x = self.fc2(x)
		return x
	
"""CNN (7 layers) with batch normalisation in:
https://github.com/Verified-Intelligence/auto_LiRPA/blob/master/examples/vision/models/feedforward.py
and
https://github.com/shizhouxing/Fast-Certified-Robust-Training/blob/main/models/feedforward.py
"""

class Flatten(nn.Module):
	def forward(self, x):
		return x.view(x.size(0), -1)
	
def CNN7(in_ch=3, in_dim=32, width=64, linear_size=512, num_class=10):
	return nn.Sequential(
		nn.Conv2d(in_ch, width, 3, stride=1, padding=1),
		nn.BatchNorm2d(width),
		nn.ReLU(),
		nn.Conv2d(width, width, 3, stride=1, padding=1),
		nn.BatchNorm2d(width),
		nn.ReLU(),
		nn.Conv2d(width, 2 * width, 3, stride=2, padding=1),
		nn.BatchNorm2d(2 * width),
		nn.ReLU(),
		nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
		nn.BatchNorm2d(2 * width),
		nn.ReLU(),
		nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
		nn.BatchNorm2d(2 * width),
		nn.ReLU(),
		Flatten(),
		nn.Linear((in_dim//2) * (in_dim//2) * 2 * width, linear_size),
		nn.BatchNorm1d(linear_size),
		nn.ReLU(),
		nn.Linear(linear_size,num_class)
	)

"""Resnet implementation is based on the implementation found in:
https://github.com/YisenWang/MART/blob/master/resnet.py
and
https://github.com/yaodongyu/TRADES/blob/master/models/resnet.py
"""

class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, in_planes, planes, stride=1):
		super(BasicBlock, self).__init__()
		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)

		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != self.expansion * planes:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(self.expansion * planes)
			)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		out += self.shortcut(x)
		out = F.relu(out)
		return out


class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, in_planes, planes, stride=1):
		super(Bottleneck, self).__init__()
		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(self.expansion * planes)

		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != self.expansion * planes:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(self.expansion * planes)
			)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = F.relu(self.bn2(self.conv2(out)))
		out = self.bn3(self.conv3(out))
		out += self.shortcut(x)
		out = F.relu(out)
		return out


class ResNet(nn.Module):
	def __init__(self, block, num_blocks, num_classes=10):
		super(ResNet, self).__init__()
		self.in_planes = 64

		self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
		self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
		self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
		self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
		self.linear = nn.Linear(512 * block.expansion, num_classes)

	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1] * (num_blocks - 1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes * block.expansion
		return nn.Sequential(*layers)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = F.avg_pool2d(out, 4)
		out = out.view(out.size(0), -1)
		out = self.linear(out)
		return out


def ResNet18():
	return ResNet(BasicBlock, [2, 2, 2, 2])

def ResNet50():
	return ResNet(Bottleneck, [3, 4, 6, 3])
