from functools import partial
from collections.abc import Callable
import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import nn
from module.MLP import MLP

class MLPModel(
  nn.Module
):
  def __init__(
      self,
      input_dim: int,
      hidden_dim: int,
      output_dim: int,
      num_layers: int,
      activation: Callable[[torch.Tensor], torch.Tensor] = F.relu,
      dropout: float=0.1,
      bias: bool=True,
      dtype: torch.dtype=torch.float32
  ):
    super().__init__()
    self.dtype = dtype
    MLP_obj = partial(
      MLP,
      input_dim=input_dim,
      hidden_dim=hidden_dim,
      output_dim=output_dim,
      num_layers=num_layers,
      bias=bias,
      dtype=self.dtype
    )
    self.MLP = MLP_obj(activation=activation, dropout=dropout)

  def forward(
      self,
      x: Float[torch.Tensor, "*batch input_dim"],
  ):
    if x.dtype != self.dtype or x.device != self.MLP.device:
      x = x.to(dtype=self.dtype, device=self.MLP.device)
    
    return self.MLP(x)