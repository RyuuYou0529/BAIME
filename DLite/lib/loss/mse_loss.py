import torch

from .base_loss import BaseLoss
from ..core.register import LOSS_REGISTER

@LOSS_REGISTER.register('MSE')
class MSELoss(BaseLoss):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.mse_loss_fn = torch.nn.MSELoss(reduction='mean')
        self.loss_logger = dict()
        self.image_logger = dict()

    @classmethod
    def init_from_args(cls, args):
        instance = cls()
        return instance

    def forward(self, preds, labels):
        loss = self.mse_loss_fn(preds, labels)
        self.loss_logger['MSE_loss'] = loss.item()
        self.image_logger['Predictions'] = preds

        return loss, self.loss_logger, self.image_logger