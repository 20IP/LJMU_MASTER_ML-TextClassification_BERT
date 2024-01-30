import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class CrossEntropyLoss(nn.Module):
    ''' 
    Cross Entropy Loss
    
    This class defines the Cross Entropy Loss for classification tasks.
    It uses PyTorch's built-in CrossEntropyLoss.

    Attributes:
        loss_fn (torch.nn.CrossEntropyLoss): PyTorch Cross Entropy Loss instance.
    '''

    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        ''' 
        Forward pass for Cross Entropy Loss.

        Args:
            logits (torch.Tensor): Logits predicted by the model.
            labels (torch.Tensor): True labels.

        Returns:
            torch.Tensor: Computed Cross Entropy Loss.
        '''
        return self.loss_fn(logits, labels)

class FocalLoss(nn.Module):
    ''' 
    Focal Loss
    
    This class defines the Focal Loss for addressing class imbalance in classification tasks.
    It introduces a modulating factor (gamma) to down-weight easy samples.

    Attributes:
        gamma (float): Modulating factor for Focal Loss.
    '''

    def __init__(self, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, outputs, labels):
        ''' 
        Forward pass for Focal Loss.

        Args:
            outputs (torch.Tensor): Raw outputs from the model.
            labels (torch.Tensor): True labels.

        Returns:
            torch.Tensor: Computed Focal Loss.
        '''
        ce_loss = F.cross_entropy(outputs, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = (1 - pt) ** self.gamma * ce_loss
        return loss.mean()

class FocalLossWithBatchNormL2(nn.Module):
    ''' 
    Focal Loss with BatchNorm L2 Penalty
    
    This class defines Focal Loss with an additional BatchNorm L2 penalty.
    It helps prevent overfitting by penalizing large weights in BatchNorm layers.

    Attributes:
        gamma (float): Modulating factor for Focal Loss.
        beta (float): Coefficient for BatchNorm L2 penalty.
    '''

    def __init__(self, gamma=2.0, beta=1e-4):
        super(FocalLossWithBatchNormL2, self).__init__()
        self.gamma = gamma
        self.beta = beta

    def forward(self, outputs, labels):
        ''' 
        Forward pass for Focal Loss with BatchNorm L2 Penalty.

        Args:
            outputs (torch.Tensor): Raw outputs from the model.
            labels (torch.Tensor): True labels.

        Returns:
            torch.Tensor: Computed Focal Loss with BatchNorm L2 Penalty.
        '''
        ce_loss = F.cross_entropy(outputs, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = (1 - pt) ** self.gamma * ce_loss
        return loss.mean() + self.beta * self.batch_norm_l2_penalty()

    def batch_norm_l2_penalty(self):
        ''' 
        Compute BatchNorm L2 Penalty.

        Returns:
            torch.Tensor: L2 penalty for BatchNorm layers.
        '''
        l2_penalty = torch.tensor(0.0, requires_grad=True)
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                l2_penalty += (module.weight ** 2).sum()
        return l2_penalty

class LabelSmoothingLoss(nn.Module):
    ''' 
    Label Smoothing Loss
    
    This class defines the Label Smoothing Loss for classification tasks.
    It mitigates overconfidence in the model predictions by introducing label smoothing.

    Attributes:
        smoothing (float): Smoothing factor for label smoothing.
    '''

    def __init__(self, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, outputs, labels):
        ''' 
        Forward pass for Label Smoothing Loss.

        Args:
            outputs (torch.Tensor): Logits predicted by the model.
            labels (torch.Tensor): True labels.

        Returns:
            torch.Tensor: Computed Label Smoothing Loss.
        '''
        log_probs = F.log_softmax(outputs, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
