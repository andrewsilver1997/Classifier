import torch.nn as nn
from torchvision.models import alexnet, resnet50, densenet121, densenet169, densenet201, densenet161


class Classifier(nn.Module):
    def __init__(self, model):
        super(Classifier, self).__init__()
        self.model = model
        
    def build_model(self):
        if self.model == 'alexnet':
            model = alexnet(pretrained=True)
        elif self.model == 'resnet50':
            model = resnet50(pretrained=True)
        elif self.model == 'densenet121':
            model = densenet121(pretrained=True)
        elif self.model == 'densenet169':
            model = densenet169(pretrained=True)
        elif self.model == 'densenet201':
            model = densenet201(pretrained=True)
        elif self.model == 'densenet161':
            model = densenet161()
        return model

