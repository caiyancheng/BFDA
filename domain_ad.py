import torch
import torch.nn as nn

# def get_fc_discriminator(num_classes, ndf=128):
#     return nn.Sequential(
#         nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1),
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1),
#     )

def bce_loss(y_pred, y_label):
    # y_truth_tensor = torch.FloatTensor(y_pred.size())
    # y_truth_tensor.fill_(y_label)
    # y_truth_tensor = y_truth_tensor.to(y_pred.get_device())
    y_truth_tensor = torch.FloatTensor(y_pred.size())
    y_truth_tensor.fill_(0)
    try:
        y_truth_tensor[:,y_label] = 1.
    except:
        x = 1
    y_truth_tensor = y_truth_tensor.to(y_pred.get_device())
    return nn.BCEWithLogitsLoss()(y_pred, y_truth_tensor)
    # return nn.BCELoss()(y_pred, y_truth_tensor)

def ls_loss(y_pred, y_label,reduction='mean'):
    y_truth_tensor = torch.FloatTensor(y_pred.size())
    y_truth_tensor.fill_(y_label)
    y_truth_tensor = y_truth_tensor.to(y_pred.get_device())
    return nn.MSELoss(reduction=reduction)(y_pred, y_truth_tensor)
