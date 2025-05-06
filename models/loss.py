import torch.nn as nn
from torch.nn import functional as F
from torchvision.ops.focal_loss import sigmoid_focal_loss
import torch


def get_loss_module(config):

    loss_type = config.loss

    if loss_type == "cross_entropy":
        return NoFussCrossEntropyLoss(
            reduction="none"
        )  # outputs loss for each batch sample

    if loss_type == "mae":
        return nn.L1Loss(reduction="none")

    if loss_type == "mse":
        return nn.MSELoss(reduction="none")

    else:
        raise ValueError(f"Loss module for '{loss_type}' does not exist")


class NoFussCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    pytorch's CrossEntropyLoss is fussy: 1) needs Long (int64) targets only, and 2) only 1D.
    This function satisfies these requirements
    """

    def forward(self, inp, target):
        return F.cross_entropy(
            inp,
            target.long().squeeze(),
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )
