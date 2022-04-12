# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
from PIL import Image

import paddle
import paddle.nn as nn
from paddle.vision import transforms

from models.resnet_csra import ResNet_CSRA
from utils.evaluation.eval import class_dict


# Usage:
# This demo is used to predict the label of each image
# if you want to use our models to predict some labels of the VOC2007 images
# 1st: use the models pretrained on VOC2007
# 2nd: put the images in the utils/demo_images
# 3rd: run predict.py

def Args():
    parser = argparse.ArgumentParser(description="settings")
    # model default resnet101
    parser.add_argument("--model", default="resnet101", type=str)
    parser.add_argument("--num_heads", default=1, type=int)
    parser.add_argument("--lam",default=0.1, type=float)
    parser.add_argument("--load_from", default="models_local/resnet101_voc07_head1_lam0.1_94.7.pth", type=str)
    parser.add_argument("--img_dir", default="images/", type=str)

    # dataset
    parser.add_argument("--dataset", default="voc07", type=str)
    parser.add_argument("--num_cls", default=20, type=int)

    args = parser.parse_args()
    return args
    

def demo():
    args = Args()

    # model 
    if args.model == "resnet101": 
        model = ResNet_CSRA(num_heads=args.num_heads, lam=args.lam, num_classes=args.num_cls, depth=101, pretrained=False)
        normalize = transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
        img_size = 448

    model.load_dict(paddle.load(args.load_from))
    print("Loading weights from {}".format(args.load_from))

    # image pre-process
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])

    # prediction of each image's label
    for img_file in os.listdir(args.img_dir):
        print(os.path.join(args.img_dir, img_file), end=" prediction: ")
        img = Image.open(os.path.join(args.img_dir, img_file)).convert("RGB")
        img = transform(img)
        img = img.unsqueeze(0)

        model.eval()
        logit = model(img).squeeze(0)
        logit = nn.Sigmoid()(logit)


        # pos = paddle.where(logit > 0.5)[0].cpu().numpy()
        pos = []
        for i in range(len(logit)):
            if logit[i] > 0.5:
                pos.append(i)
        for k in pos:
            print(class_dict[args.dataset][k], end=",")
        print()


if __name__ == "__main__":
    demo()
