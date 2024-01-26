import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, outputs, labels):
        ce_loss = F.cross_entropy(outputs, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = (1 - pt) ** self.gamma * ce_loss
        return loss.mean()


class FocalLossWithBatchNormL2(nn.Module):
    def __init__(self, gamma=2.0, beta=1e-4):
        super(FocalLossWithBatchNormL2, self).__init__()
        self.gamma = gamma
        self.beta = beta

    def forward(self, outputs, labels):
        ce_loss = F.cross_entropy(outputs, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = (1 - pt) ** self.gamma * ce_loss
        return loss.mean() + self.beta * self.batch_norm_l2_penalty()

    def batch_norm_l2_penalty(self):
        l2_penalty = torch.tensor(0.0, requires_grad=True)
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                l2_penalty += (module.weight ** 2).sum()
        return l2_penalty


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, outputs, labels):
        log_probs = F.log_softmax(outputs, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, outputs, labels):
        log_probs = F.log_softmax(outputs, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
