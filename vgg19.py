from vgg_blocks import *

import torch
import torch.nn as nn

class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        # input: 224x224 RGB images
        self.double_conv_1 = DoubleConv(3, 64)
        self.double_conv_2 = DoubleConv(64, 128)

        self.fourth_conv_1 = FourthConv(128, 256)
        self.fourth_conv_2 = FourthConv(256, 512)
        self.fourth_conv_3 = FourthConv(512, 512)

        self.fl_1 = FullyConnectedLayer(25088, 4096, first_l=True, final_l=False)
        self.fl_2 = FullyConnectedLayer(4096, 4096, first_l=False, final_l=False)
        self.fl_3 = FullyConnectedLayer(4096, 10, first_l=False, final_l=True)

    def forward(self, x):
        x1 = self.double_conv_1(x)
        x2 = self.double_conv_2(x1)

        x3 = self.fourth_conv_1(x2)
        x4 = self.fourth_conv_2(x3)
        x5 = self.fourth_conv_3(x4)

        # input to linear function is x5.size([1, 512, 7, 7]) -> 512*7*7=25088
        x6 = self.fl_1(x5)
        x7 = self.fl_2(x6)
        x8 = self.fl_3(x7)
        return x8

"""
if __name__ == "__main__":
    model = VGG19().cuda()

    image = torch.rand((1, 3, 224, 224)).to(torch.device("cuda:0"))
    output = model(image)
"""
