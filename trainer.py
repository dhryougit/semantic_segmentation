import copy
import time

import torch
import torch.nn.functional as F

from losses import dice_loss_with_logits
from metric import AverageMeterSet


def print_metrics(phase, average_meter_set: AverageMeterSet):
    results = ["{}: {:4f}".format(k, v) for k, v in average_meter_set.averages().items()]
    print("{}: {}".format(phase, ", ".join(results)))


def train_one_epoch(model, dataloader, optimizer, device, bce_weight=0.5):
    average_meter_set = AverageMeterSet()
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        targets = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        logits = model(inputs)

        bce_loss = F.binary_cross_entropy_with_logits(logits, targets)
        dice_loss = dice_loss_with_logits(logits, targets)

        loss = bce_loss * bce_weight + dice_loss * (1 - bce_weight)

        loss.backward()
        optimizer.step()

        average_meter_set.update('bce_loss', bce_loss.item())
        average_meter_set.update('dice_loss', dice_loss.item())
        average_meter_set.update('loss', loss.item())

    return average_meter_set, model


def validate(model, dataloader, device, bce_weight=0.5):
    average_meter_set = AverageMeterSet()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            targets = labels.to(device)

            logits = model(inputs)

            bce_loss = F.binary_cross_entropy_with_logits(logits, targets)
            dice_loss = dice_loss_with_logits(logits, targets)

            loss = bce_loss * bce_weight + dice_loss * (1 - bce_weight)

            average_meter_set.update('bce_loss', bce_loss.item())
            average_meter_set.update('dice_loss', dice_loss.item())
            average_meter_set.update('loss', loss.item())

    return average_meter_set


def train_model(model, dataloaders, optimizer, lr_scheduler, device, bce_weight=0.5, num_epochs=25):
    model = model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                average_meter_set, model = train_one_epoch(model, dataloaders[phase], optimizer, device, bce_weight)
                print_metrics(phase, average_meter_set)
                lr_scheduler.step()
            else:
                model.eval()  # Set model to evaluate mode
                average_meter_set = validate(model, dataloaders[phase], device, bce_weight)
                print_metrics(phase, average_meter_set)
                epoch_loss = average_meter_set.averages()['loss']

                if epoch_loss < best_loss:
                    print("Saving the Best Model")
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
