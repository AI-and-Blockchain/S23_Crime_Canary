import os 
import torch
from PIL import Image
from torch.nn import Module
from model import SiameseNetwork
from torchvision import transforms
from typing import Iterable, List
from model import SiameseNetwork, get_default_accident_model, get_default_severity_model


DEFAULT_SHAPE = [3, 28, 28]

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])


def resize_image(image: Image) -> torch.Tensor:
    return transform(image).unsqueeze(0)


def get_models(model_paths: List[str]) -> Iterable[Module]:
    assert(len(model_paths) == 2)
    
    for path in model_paths:
        assert (os.path.exists(path))
    
    accident, severity = get_default_accident_model(), get_default_severity_model()
    
    return [torch.load(path).eval() for path in model_paths]


def predict(image: torch.Tensor, models: Iterable[Module]) -> Iterable[torch.Tensor]:
    assert (list(image.shape[1:]) == DEFAULT_SHAPE)
    
    accident = models[0]
    severity = models[1]
    
    alogits = accident(image, image)[1]
    slogits = severity(image, image)[1]
    
    apred, spred = map(
        lambda x : torch.argmax(alogits, dim=-1).squeeze().item(), (alogits, slogits))
    
    return apred, spred