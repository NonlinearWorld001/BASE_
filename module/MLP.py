import torch
import torch.nn as nn
import torch.nn.functional as F

from collections.abc import Callable
from typing import Optional
from jaxtyping import Float


class MLP(nn.Module):
  def __init__(
      self,
      input_dim: int,
      hidden_dim: int,
      output_dim: int,
      num_layers: int,
      *,
      activation: Callable[[torch.Tensor], torch.Tensor] = F.relu,
      dropout: float=0.1,
      bias: bool=True,
      dtype: torch.dtype = torch.float32
  ):
    super().__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.output_dim = output_dim
    self.num_layers = num_layers
    self.bias = bias
    self.dropout = dropout
    self.activation = activation
    self.dtype = dtype

    self.in_proj = nn.Linear(input_dim, hidden_dim, bias=self.bias, dtype=self.dtype)
    self.hidden_layers = nn.ModuleList(
      [
        nn.Linear(hidden_dim, hidden_dim, bias=self.bias, dtype=self.dtype)
        for _ in range(num_layers - 1)
      ]
    )
    self.out_proj = nn.Linear(hidden_dim, output_dim, bias=self.bias, dtype=self.dtype)
    
    self.dropout_layer = nn.Dropout(self.dropout)

    self._initialize_weights('xavier')

  def _initialize_weights(self, method: str = 'xavier'):
    if method not in ['xavier', 'he']:
            raise ValueError(f"only suuupport 'xavier'/'he', now is: {method}")
    
    for module in self.modules():
        if isinstance(module, nn.Linear):
            if method == 'xavier':
                nn.init.xavier_uniform_(module.weight)
            elif method == 'he':
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
  
  def forward(
      self,
      x: Float[torch.Tensor, "... input_dim"],
  )-> Float[torch.Tensor, "... output_dim"]:
    if x.dtype != self.dtype:
            x = x.to(self.dtype)
    x = self.in_proj(x)
    if self.activation is not None:  
            x = self.activation(x)
    x = self.dropout_layer(x)
    for layer in self.hidden_layers:
      x = layer(x)
      x = self.activation(x)
      x = self.dropout_layer(x)
    output = self.out_proj(x)
    return output

    
    