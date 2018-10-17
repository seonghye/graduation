from __future__ import print_function
import torch
from torch.autograd import Variable
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dataset
from operator import itemgetter
import tensorly as tl
import tensorly
from itertools import chain
from decompositions import cp_decomposition_conv_layer, tucker_decomposition_conv_layer
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
#from data import VOC_ROOT, VOCAnnotationTransform, VOCDetection, BaseTransform
#from data import VOC_CLASSES as labelmap
import torch.utils.data as data
import torchvision
from ssd import build_ss
import sys
import os
import time
import argparse
import numpy as np
import pickle
import cv2

def decompsose(model):
    N = len(model)
    for i in range(0,N):
        if isinstance(model[i], nn.modules.conv.Conv2d):
            conv_layer = model[i]
            decomposed = tucker_decomposition_conv_layer(conv_layer)
            model[i] = decomposed
    return model






if sys.version_info[0] ==2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

num_classes = len(labelmap)+1
net = build_ssd('test', 300, num_classes)
net.load_state_dict(torch.load('./ssd300_COCO_30000.pth'))
net.eval()

print("Net:", net)
print("Net.vgg:", net.vgg)
print("Net.extras:",net.extras)
print("Loc:", net.loc)
print("Conf:", net.conf)


model = net.vgg.cuda()
model = decompose(model)
print(model)

