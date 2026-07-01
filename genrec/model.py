

from typing import Any, Dict

from genrec.dataset import AbstractDataset
from genrec.tokenizer import AbstractTokenizer

from torch import nn


class AbstractModel(nn.Module):
  """Abstract class for GenRec models.
  Attributes:
    config: A dictionary containing the model configuration.
    dataset: The dataset used to train the model.
    tokenizer: The tokenizer used to process the data.
    n_parameters: The total number of trainable parameters in the model.
  """

  def __init__(
      self,
      config: Dict[str, Any],
      dataset: AbstractDataset,
      tokenizer: AbstractTokenizer,
  ):
    super().__init__()

    self.config = config
    self.dataset = dataset
    self.tokenizer = tokenizer

  @property
  def n_parameters(self) -> str:
    """Calculates the total number of trainable parameters in the model."""
    total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    return f'Total number of trainable parameters: {total_params}'

  def calculate_loss(self, batch):
    raise NotImplementedError('calculate_loss method must be implemented.')

  def generate(self, batch, n_return_sequences=1):
    raise NotImplementedError('predict method must be implemented.')
