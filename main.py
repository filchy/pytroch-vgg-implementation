from vgg13 import *
from vgg16 import *
from vgg19 import *

import sys
import torch

if __name__ == "__main__":
    vgg_type = str(sys.argv[1])

    if vgg_type not in ["vgg13", "vgg16","vgg19"]:
        raise Exception("Wrong VGG type was given. [vgg13, vgg16, vgg19]")

    if vgg_type == "vgg13":
        model = VGG13()
    elif vgg_type == "vgg16":
        model = VGG16()
    else:
        model = VGG19()

    if torch.cuda.is_available():
        model = model.cuda()

        image = torch.rand((1, 3, 224, 224)).to(torch.device("cuda:0"))
        output = model(image)
    else:
        image = torch.rand((1, 3, 224, 224))
        output = model(image)

    print(output)
