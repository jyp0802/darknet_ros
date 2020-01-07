#!/usr/bin/env python
import rospy
import rospkg

import cv2
from cv_bridge import CvBridge, CvBridgeError
import torch
from torch import nn
from torchvision import transforms

import datetime

from darknet_ros.srv import *

class CustomConvNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomConvNet, self).__init__()

        self.layer1 = self.conv_module(3, 16)
        self.layer2 = self.conv_module(16, 32)
        self.layer3 = self.conv_module(32, 64)
        self.layer4 = self.conv_module(64, 128)
        self.layer5 = self.conv_module(128, 256)
        self.gap = self.global_avg_pool(256, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.gap(out)
        out = out.view(-1, num_classes)

        return out

    def conv_module(self, in_num, out_num):
        return nn.Sequential(
            nn.Conv2d(in_num, out_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_num),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

    def global_avg_pool(self, in_num, out_num):
        return nn.Sequential(
            nn.Conv2d(in_num, out_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_num),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1, 1)))

def classify(img):
    global image_name, save_time
    # if (datetime.datetime.now()-save_time).seconds > 0.3:
    # cv2.imwrite(path+"/scripts/img_ros/1_" + str(image_name) + ".png", img)
    # image_name += 1
        # save_time = datetime.datetime.now()
    with torch.no_grad():
        img = cv2.resize(img, (128, 128))
        img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).to(device)
        img = img.unsqueeze(0)

        outputs = custom_model(img)
        _, predicted = torch.max(outputs.data, 1)
        return predicted.numpy()[0] == 0

def call_classify(req):
    bridge = CvBridge()
    try:
        img = bridge.imgmsg_to_cv2(req.input, "rgb8")
    except CvBridgeError as e:
        print(e)

    out = classify(img)
    return ClassifyImageResponse(out)

rospack = rospkg.RosPack()
path = rospack.get_path('darknet_ros')
# image_name = 0
# save_time = datetime.datetime.now()

rospy.init_node('book_classifier_server')

num_classes = 2

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

custom_model = CustomConvNet(num_classes=num_classes).to(device)
custom_model.load_state_dict(torch.load(path+"/scripts/model_3.pth"))

image_transforms = transforms.Compose([transforms.Resize((128, 128)),
                                       transforms.ToTensor()])

custom_model.eval()

s = rospy.Service('darknet_ros/book_classifier', ClassifyImage, call_classify)
rospy.loginfo("Starting book_classifier_server")

rospy.spin()