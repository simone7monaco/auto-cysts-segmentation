from pytorch_toolbelt.losses import JaccardLoss, BinaryFocalLoss, BinarySoftF1Loss
from torch import nn


class ComboLoss(nn.Module):
    def __init__(self, weights=[0.1, 0.9], losses=["jaccard", "f1"]):
        super(ComboLoss, self).__init__()
        losses_dict = {
            "jaccard": JaccardLoss(mode="binary", from_logits=True),
            "f1": BinarySoftF1Loss(),
            "bce": nn.BCEWithLogitsLoss()   

        }

        self.losses = [(n, w, losses_dict.get(n)) for n, w in zip(losses, weights)]

    def forward(self, logits, masks):
        total_loss = 0
        for loss_name, weight, loss in self.losses:
            if loss is not None:
                ls_mask = loss(logits, masks)
                total_loss += weight * ls_mask
        return total_loss
