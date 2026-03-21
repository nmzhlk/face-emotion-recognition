from typing import Any, List, Optional

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

EMOTION_LABELS: List[str] = [
    "Anger",
    "Contempt",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise",
]


def get_resnet_emotion_model(
    output_shape: int = 7, load_path: Optional[str] = None
) -> Any:
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    pretrained_conv1 = model.conv1.weight
    model.conv1 = nn.Conv2d(
        1, model.inplanes, kernel_size=7, stride=2, padding=3, bias=False
    )
    model.conv1.weight.data = pretrained_conv1.mean(dim=1, keepdim=True)

    model.fc = nn.Linear(model.fc.in_features, output_shape)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.conv1.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

    if load_path:
        state_dict = torch.load(load_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)

    return model
