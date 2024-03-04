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
        logits_sigmoid = torch.sigmoid(logits)
        logits_flat = logits_sigmoid.view(-1)
        labels_flat = labels.view(-1)
        
        # loss = -((1 - logits_flat) ** self.gamma) * labels_flat * torch.log(logits_flat) - (1 - labels_flat) * (logits_flat ** self.gamma) * torch.log(1 - logits_flat)
        pos_loss = -((1 - logits_flat) ** self.gamma) * labels_flat * torch.log(logits_flat)
        neg_loss = -((logits_flat) ** self.gamma) * (1 - labels_flat) * torch.log(1 - logits_flat)
        loss = pos_loss + neg_loss
    
        return loss.mean()
    
class LabelSmoothingLossMultiLabel(nn.Module):
    ''' 
    Label Smoothing Loss for Multi-Label Classification
    
    This class defines the Label Smoothing Loss for addressing multi-label classification tasks.
    It mitigates overconfidence in the model predictions by introducing label smoothing.

    Attributes:
        smoothing (float): Smoothing factor for label smoothing.
    '''

    def __init__(self, epsilon=0.3, gamma=2):
        super(LabelSmoothingLossMultiLabel, self).__init__()
        self.epsilon = epsilon
        self.gamma = gamma

    def forward(self, logits, labels):
        ''' 
        Forward pass for Label Smoothing + Focal Loss.

        Args:
            logits (torch.Tensor): Logits predicted by the model.
            labels (torch.Tensor): True labels.

        Returns:
            torch.Tensor: Computed Label Smoothing + Focal Loss.
        '''


        sm_labels = (1 - self.epsilon) * labels + (self.epsilon / 5)
        print('sm_labels: ', sm_labels)

        logits_sigmoid = torch.sigmoid(logits)
        log_probs = torch.log(logits_sigmoid)
        logits_flat = logits_sigmoid.view(-1)
        labels_flat = sm_labels.view(-1)

        # Focal Loss components
        pos_loss = -((1 - logits_flat) ** self.gamma) * labels_flat * torch.log(logits_flat)
        neg_loss = -((logits_flat) ** self.gamma) * (1 - labels_flat) * torch.log(1 - logits_flat)

        # Combine Focal Loss components
        loss = pos_loss + neg_loss

        return loss.mean()