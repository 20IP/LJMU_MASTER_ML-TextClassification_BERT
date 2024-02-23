import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class CrossEntropyLossMultiLabel(nn.Module):
    ''' 
    Cross Entropy Loss for Multi-Label Classification
    
    This class defines the Cross Entropy Loss for addressing multi-label classification tasks.
    It uses PyTorch's built-in CrossEntropyLoss, adjusted for multi-label.

    Attributes:
        loss_fn (torch.nn.CrossEntropyLoss): PyTorch Cross Entropy Loss instance.
    '''

    def __init__(self):
        super(CrossEntropyLossMultiLabel, self).__init__()

    def forward(self, logits, labels):
        ''' 
        Forward pass for Cross Entropy Loss.

        Args:
            logits (torch.Tensor): Logits predicted by the model.
            labels (torch.Tensor): True labels.

        Returns:
            torch.Tensor: Computed Cross Entropy Loss.
        '''
        # Apply sigmoid activation to logits for multi-label classification
        logits_sigmoid = torch.sigmoid(logits)
        # logits_sigmoid = (logits_sigmoid >= 0.5).float()

        # Flatten the logits and labels for multi-label loss calculation
        logits_flat = logits_sigmoid.view(-1)
        labels_flat = labels.view(-1)

        # Binary cross entropy loss
        loss = F.binary_cross_entropy(logits_flat, labels_flat)

        return loss

    
class FocalLossMultiLabel(nn.Module):
    ''' 
    Focal Loss for Multi-Label Classification
    
    This class defines the Focal Loss for addressing class imbalance in multi-label classification tasks.
    It introduces a modulating factor (gamma) to down-weight easy samples.

    Attributes:
        gamma (float): Modulating factor for Focal Loss.
    '''

    def __init__(self, gamma=2.0):
        super(FocalLossMultiLabel, self).__init__()
        self.gamma = gamma

    def forward(self, logits, labels):
        ''' 
        Forward pass for Focal Loss.

        Args:
            outputs (torch.Tensor): Raw outputs from the model.
            labels (torch.Tensor): True labels.

        Returns:
            torch.Tensor: Computed Focal Loss.
        '''
        ce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = (1 - pt) ** self.gamma * ce_loss
        return loss.mean()
    
class FocalLossWithBatchNormL2MultiLabel(nn.Module):
    ''' 
    Focal Loss with BatchNorm L2 Penalty for Multi-Label Classification
    
    This class defines Focal Loss with an additional BatchNorm L2 penalty for multi-label classification.
    It helps prevent overfitting by penalizing large weights in BatchNorm layers.

    Attributes:
        gamma (float): Modulating factor for Focal Loss.
        beta (float): Coefficient for BatchNorm L2 penalty.
    '''

    def __init__(self, gamma=2.0, beta=1e-4):
        super(FocalLossWithBatchNormL2MultiLabel, self).__init__()
        self.gamma = gamma
        self.beta = beta

    def forward(self, logits, labels):
        ''' 
        Forward pass for Focal Loss with BatchNorm L2 Penalty.

        Args:
            outputs (torch.Tensor): Raw outputs from the model.
            labels (torch.Tensor): True labels.

        Returns:
            torch.Tensor: Computed Focal Loss with BatchNorm L2 Penalty.
        '''
        ce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
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
    
class LabelSmoothingLossMultiLabel(nn.Module):
    ''' 
    Label Smoothing Loss for Multi-Label Classification
    
    This class defines the Label Smoothing Loss for addressing multi-label classification tasks.
    It mitigates overconfidence in the model predictions by introducing label smoothing.

    Attributes:
        smoothing (float): Smoothing factor for label smoothing.
    '''

    def __init__(self, smoothing=0.1):
        super(LabelSmoothingLossMultiLabel, self).__init__()
        self.smoothing = smoothing

    def forward(self, logits, labels):
        ''' 
        Forward pass for Label Smoothing Loss.

        Args:
            outputs (torch.Tensor): Logits predicted by the model.
            labels (torch.Tensor): True labels.

        Returns:
            torch.Tensor: Computed Label Smoothing Loss.
        '''
        sigmoid_logits = torch.sigmoid(logits)

        smooth_labels = (1.0 - self.smoothing) * labels + self.smoothing / 2.0
        log_probs = torch.log(sigmoid_logits)

        loss = -torch.sum(smooth_labels * log_probs + (1.0 - smooth_labels) * torch.log(1.0 - sigmoid_logits))
        return loss / logits.size(0)  # Normalize by batch size