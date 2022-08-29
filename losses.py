import torch


def dice_loss_with_logits(logits: torch.Tensor, target: torch.Tensor, smooth=1.):
    pred = torch.softmax(logits, dim=1)
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()
