import torchvision.models as models

class StyleModel:
    model = models.resnet18(pretrained=True, progress=True)