import torch.nn as nn

def get_adversarial_loss():
    return nn.BCELoss()
