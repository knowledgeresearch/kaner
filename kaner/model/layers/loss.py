# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Loss Functions"""

from typing import List

import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    r"""
    The implementation of the paper `Focal Loss for Dense Object Detection`.
        Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

    References:
        [1] https://zhuanlan.zhihu.com/p/28527749
        [2] https://zhuanlan.zhihu.com/p/75542467

    Args:
        alpha (List[float]): The scalar factor for this criterion. Expecter shape is num_classes.
        gamma (float): The focusing parameter with range >= 0 that puts more focus on hard, misclassiﬁed examples.
    """

    def __init__(self, alpha: List[float] = None, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        if alpha is not None:
            self.alpha = nn.Parameter(torch.tensor(alpha).view(-1, 1))
        else:
            self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        """
        Compute focal loss.

        Args:
            inputs (torch.FLoatTensor): Input with shape n * num_classes.
            targets (torch.LongTensor): Ground truth with shape n.
        """
        n, num_classes = inputs.shape
        device = inputs.device
        if self.alpha is None:
            self.alpha = nn.Parameter(torch.tensor([1.0]*num_classes, device=device).view(-1, 1))
        alpha = self.alpha[targets.view(-1)]
        probs = torch.softmax(inputs, dim=-1).clamp(min=0.0001, max=1.0)
        # (n, num_class)
        class_mask = torch.zeros((n, num_classes), device=device)
        targets = targets.view(n, 1)
        class_mask.scatter_(1, targets, 1.)
        # (n, 1)
        probs = (probs * class_mask).sum(dim=-1).view(-1, 1)
        batch_loss = -alpha * torch.pow(1 - probs, self.gamma) * torch.log(probs)
        loss = batch_loss.mean()

        return loss
