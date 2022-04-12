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

import time
import argparse
from tqdm import tqdm

import paddle
import paddle.nn as nn
import paddle.optimizer as optim
from paddle.io import DataLoader

from models.resnet_csra import ResNet_CSRA
from datasets.dataset import DataSet
from utils.evaluation.eval import evaluation


# modify for wider dataset and vit models

def Args():
    parser = argparse.ArgumentParser(description="settings")
    # model
    parser.add_argument("--model", default="resnet101")
    parser.add_argument("--num_heads", default=1, type=int)
    parser.add_argument("--lam", default=0.1, type=float)
    parser.add_argument("--save_dir", default='checkpoint', type=str)
    # dataset
    parser.add_argument("--dataset", default="voc07", type=str)
    parser.add_argument("--num_cls", default=20, type=int)
    parser.add_argument("--train_aug", default=["randomflip", "resizedcrop"], type=list)
    parser.add_argument("--test_aug", default=[], type=list)
    parser.add_argument("--img_size", default=448, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    # optimizer, default SGD
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--poly_decay", default=False, type=bool)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--w_d", default=0.0001, type=float, help="weight_decay")
    parser.add_argument("--warmup_epoch", default=2, type=int)
    parser.add_argument("--total_epoch", default=30, type=int)
    parser.add_argument("--tranval", default=True, type=bool)
    parser.add_argument("--print_freq", default=100, type=int)
    args = parser.parse_args()
    return args


def train(i, args, model, train_loader, optimizer, warmup_scheduler):
    model.train()
    epoch_begin = time.time()
    end_time = time.time()
    for index, data in enumerate(train_loader):
        batch_begin = time.time()
        data_time = batch_begin - end_time
        img = data['img']
        target = data['target']

        optimizer.clear_grad()
        logit, loss = model(img, target)
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        lr = optimizer.get_lr()
        end_time = time.time()
        t = end_time - batch_begin

        if index % args.print_freq == 0:
            print("Epoch {}[{}/{}]: loss:{:.5f}, lr:{:.5f}, batch time:{:.4f}, data time:{:.4f}".format(
                i,
                args.batch_size * (index + 1),
                len(train_loader.dataset),
                loss.cpu().numpy()[0],
                float(lr),
                float(t),
                float(data_time)
            ))
        if i <= args.warmup_epoch:
            # update lr
            if isinstance(optimizer, paddle.distributed.fleet.Fleet):
                lr_sche = optimizer.user_defined_optimizer._learning_rate
            else:
                lr_sche = optimizer._learning_rate
            if isinstance(lr_sche, paddle.optimizer.lr.LRScheduler):
                lr_sche.step()

    t = time.time() - epoch_begin
    print("Epoch {} training ends, total {:.2f}s".format(i, t))


def val(i, args, model, test_loader, test_file):
    model.eval()
    print("Test on Epoch {}".format(i))
    result_list = []

    # calculate logit
    for index, data in enumerate(tqdm(test_loader)):
        img = data['img']
        target = data['target']
        img_path = data['img_path']

        with paddle.no_grad():
            logit = model(img)

        result = nn.Sigmoid()(logit).cpu().detach().numpy().tolist()
        for k in range(len(img_path)):
            result_list.append(
                {
                    "file_name": img_path[k].split("/")[-1].split(".")[0],
                    "scores": result[k]
                }
            )
    # cal_mAP OP OR
    evaluation(result=result_list, types=args.dataset, ann_path=test_file[0])


def main():
    args = Args()

    # model
    if args.model == "resnet101":
        model = ResNet_CSRA(num_heads=args.num_heads, lam=args.lam, num_classes=args.num_cls, depth=101,
                            pretrained=True)

    if paddle.device.cuda.device_count() > 1:
        print("lets use {} GPUs.".format(paddle.device.cuda.device_count()))
        model = paddle.DataParallel(model)

    # data
    if args.dataset == "voc07":
        train_file = ["data/voc07/trainval_voc07.json"]
        test_file = ['data/voc07/test_voc07.json']
        step_size = 4
    if args.dataset == "coco":
        train_file = ['data/coco/train_coco2014.json']
        test_file = ['data/coco/val_coco2014.json']
        step_size = 5
    if args.dataset == "wider":
        train_file = ['data/wider/trainval_wider.json']
        test_file = ["data/wider/test_wider.json"]
        step_size = 5
        args.train_aug = ["randomflip"]

    train_dataset = DataSet(train_file, args.train_aug, args.img_size, args.dataset)
    test_dataset = DataSet(test_file, args.test_aug, args.img_size, args.dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    # optimizer and warmup
    backbone, classifier = [], []
    for name, param in model.named_parameters():
        if 'classifier' in name:
            classifier.append(param)
        else:
            backbone.append(param)

    iter_per_epoch = len(train_loader)
    if args.warmup_epoch > 0:
        warmup_scheduler = optim.lr.LinearWarmup(learning_rate=optim.lr.StepDecay(learning_rate=args.lr,
                                                                                  step_size=step_size,
                                                                                  gamma=0.1),
                                                 warmup_steps=iter_per_epoch * args.warmup_epoch,
                                                 start_lr=0,
                                                 end_lr=args.lr)
        if args.poly_decay:
            warmup_scheduler = optim.lr.LinearWarmup(learning_rate=optim.lr.PolynomialDecay(learning_rate=args.lr,
                                                                                            decay_steps=args.total_epoch-args.warmup_epoch,
                                                                                            end_lr=0),
                                                     warmup_steps=iter_per_epoch * args.warmup_epoch,
                                                     start_lr=0,
                                                     end_lr=args.lr)
    else:
        warmup_scheduler = optim.lr.StepDecay(learning_rate=args.lr,
                                              step_size=step_size,
                                              gamma=0.1),
        if args.poly_decay:
            warmup_scheduler = optim.lr.LinearWarmup(learning_rate=optim.lr.PolynomialDecay(learning_rate=args.lr,
                                                                                            decay_steps=args.total_epoch-args.warmup_epoch,
                                                                                            end_lr=0),
                                                     warmup_steps=iter_per_epoch * args.warmup_epoch,
                                                     start_lr=0,
                                                     end_lr=args.lr)

    optimizer = optim.Momentum(
        learning_rate=warmup_scheduler,
        parameters=[{'params': backbone, 'learning_rate': 1},
                    {'params': classifier, 'learning_rate': 10}],
        momentum=args.momentum,
        weight_decay=args.w_d)

    # training and validation
    for i in range(1, args.total_epoch + 1):
        train(i, args, model, train_loader, optimizer, warmup_scheduler)
        paddle.save(model.state_dict(), args.save_dir + "/{}/epoch_{}.pdparams".format(args.model, i))
        if args.tranval == True:
            val(i, args, model, test_loader, test_file)
        # update lr
        if isinstance(optimizer, paddle.distributed.fleet.Fleet):
            lr_sche = optimizer.user_defined_optimizer._learning_rate
        else:
            lr_sche = optimizer._learning_rate
        if isinstance(lr_sche, paddle.optimizer.lr.LRScheduler):
            lr_sche.step()


if __name__ == "__main__":
    main()