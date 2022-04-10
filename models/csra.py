import paddle
import paddle.nn as nn


class CSRA(nn.Layer): # one basic block 
    def __init__(self, input_dim, num_classes, T, lam):
        super(CSRA, self).__init__()
        self.T = T      # temperature       
        self.lam = lam  # Lambda                        
        self.head = nn.Conv2D(input_dim, num_classes, 1, bias_attr=False)
        self.softmax = nn.Softmax(axis=2)

    def forward(self, x):
        # x (B d H W)
        # normalize classifier
        # score (B C HxW)
        score = self.head(x) / paddle.norm(self.head.weight, axis=1, keepdim=True).transpose((1, 0, 2, 3))
        score = score.flatten(2)
        base_logit = paddle.mean(score, axis=2)

        if self.T == 99: # max-pooling
            att_logit = paddle.max(score, axis=2)[0]
        else:
            score_soft = self.softmax(score * self.T)
            att_logit = paddle.sum(score * score_soft, axis=2)

        return base_logit + self.lam * att_logit

    


class MHA(nn.Layer):  # multi-head attention
    temp_settings = {  # softmax temperature settings
        1: [1],
        2: [1, 99],
        4: [1, 2, 4, 99],
        6: [1, 2, 3, 4, 5, 99],
        8: [1, 2, 3, 4, 5, 6, 7, 99]
    }

    def __init__(self, num_heads, lam, input_dim, num_classes):
        super(MHA, self).__init__()
        self.temp_list = self.temp_settings[num_heads]
        self.multi_head = nn.LayerList([
            CSRA(input_dim, num_classes, self.temp_list[i], lam)
            for i in range(num_heads)
        ])

    def forward(self, x):
        logit = 0.
        for head in self.multi_head:
            logit += head(x)
        return logit
