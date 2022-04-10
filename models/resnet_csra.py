from .csra import MHA
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.vision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152


class ResNet_CSRA(nn.Layer):
    def __init__(self, num_heads, lam, num_classes, depth=101, input_dim=2048, pretrained=True):
        self.depth = depth
        super(ResNet_CSRA, self).__init__()
        self.init_weights(pretrained=pretrained)
        self.classifier = MHA(num_heads, lam, input_dim, num_classes) 
        self.loss_func = F.binary_cross_entropy_with_logits

    def backbone_forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        return x

    def forward_train(self, x, target):
        x = self.backbone_forward(x)
        logit = self.classifier(x)
        loss = self.loss_func(logit, target.astype('float32'), reduction="mean")
        return logit, loss

    def forward_test(self, x):
        x = self.backbone_forward(x)
        x = self.classifier(x)
        return x

    def forward(self, x, target=None):
        if target is not None:
            return self.forward_train(x, target)
        else:
            return self.forward_test(x)

    def init_weights(self, pretrained=True):
        print("backbone params inited by paddle official model")
        if self.depth == 18:
            self.backbone = resnet18(pretrained=pretrained)
        if self.depth == 34:
            self.backbone = resnet34(pretrained=pretrained)
        if self.depth == 50:
            self.backbone = resnet50(pretrained=pretrained)
        if self.depth == 101:
            self.backbone = resnet101(pretrained=pretrained)
        if self.depth == 152:
            self.backbone = resnet152(pretrained=pretrained)