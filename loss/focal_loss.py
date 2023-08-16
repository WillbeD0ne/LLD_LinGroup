import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets, lam=None):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        if lam is not None:
            class_mask = lam * F.one_hot(targets[0], num_classes=C) + (1 - lam) * F.one_hot(targets[1], num_classes=C)
        else:
            class_mask = inputs.data.new(N, C).fill_(0)
            class_mask = Variable(class_mask)
            ids = targets.view(-1, 1) # [N, 1]
            class_mask.scatter_(1, ids.data, 1.) # onehot [N, 7]
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        # alpha = self.alpha[ids.data.view(-1)]
        alpha = Variable(torch.ones(N, 1)).cuda()

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

class FocalLossWithSmoothing(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, label_smoothing=0.1, alpha=None, gamma=2, size_average=True):
        super(FocalLossWithSmoothing, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):

        pos, neg = 1. - self.label_smoothing, self.label_smoothing / (self.class_num - 1)

        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(neg)
        class_mask = Variable(class_mask)
        class_mask.scatter_(1, targets.unsqueeze(1), pos).detach() # onehot [N, 7]
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[targets.data.view(-1)]

        pos_ind = class_mask.argmax(1)
        neg_ind = [torch.cat([torch.arange(0, ind), torch.arange(ind+1, 7)], dim=0).unsqueeze(0) for i, ind in enumerate(pos_ind)]
        neg_ind = torch.cat(neg_ind, dim=0)

        P_pos = [P[i, ind].unsqueeze(0) for i, ind in enumerate(pos_ind)]
        P_pos = torch.cat(P_pos, dim=0)
        P_neg = [torch.cat([P[i, :ind], P[i, ind+1:]], dim=0).unsqueeze(0) for i, ind in enumerate(pos_ind)]
        P_neg = torch.cat(P_neg, dim=0)

        pos_loss = pos * (1.-P_pos)**self.gamma * P_pos.log()
        neg_loss = (neg * (P_neg)**self.gamma * (1.-P_neg).log()).sum(1)
        batch_loss = -(pos_loss + neg_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss