#Transfer learning via Resnet50
import torchvision
import torch
import torch.nn as nn

class Resnet50(nn.Module):
    def __init__(self, Pretrained=False):
        super(Resnet50, self).__init__()
        self.model = torchvision.models.resnet50(pretrained=Pretrained)

        for param in self.model.parameters():
            param.requires_grad = False

        final_in = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(final_in, final_in // 2),
            nn.ReLU(),
            nn.Linear(final_in // 2, 2)
        )
        

