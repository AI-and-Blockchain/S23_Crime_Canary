import torch
import torch.nn as nn
from functorch import vmap
from functools import partial

from typing import Tuple, Iterable

class Siamese(nn.Module):
    def __init__(self, layers: Iterable, out_channels: int):
        super().__init__()
        
        # pass in a NN
        self.layers = vmap(nn.Sequential(*layers))
        
        # p can be change to induced norm (i.e., p = 1, p = 2)
        self.norm_fn = vmap(
            partial(torch.norm, p = 'fro')
        )
        
        # final linear layer to get prediction representation
        self.predictor = vmap(nn.Linear(1, out_channels))
    
    def forward(self, x_query: torch.Tensor, x_support: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward Pass:
            arg: Query Image (any image)
                 Support Image
        
        """
        x_query = self.layers(x_query)
        x_support = self.layers(x_support)        
        
        similarity = (self.norm_fn(x_query - x_support) + 1e-8)[:, None]
        logits = self.predictor(similarity)
        
        # Similarity : B x 1
        # Digits: B x D where D is out channels
        return similarity, logits
        

layers_example = [
    nn.Conv2d(3, 10, 1),
    nn.Conv2d(10, 10, 5),
    nn.Conv2d(10, 10, 3),
]

model = Siamese(layers_example, 10)

image_s = torch.rand([10, 3, 28, 28])
image_q = torch.rand([10, 3, 28, 28])

print(model.forward(image_q, image_s))