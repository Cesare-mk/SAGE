import logging
from typing import Any

from genrec import utils_log
from genrec import utils


class AbstractTokenizer:
  """Abstract tokenizer class for GenRec."""

  def __init__(self, config: dict[Any, Any]):
    self.config = config
    self.logger = logging.getLogger()
    self.eos_token = None
    self.collate_fn = {'train': None, 'val': None, 'test': None}

  def _init_tokenizer(self):
    raise NotImplementedError('Tokenizer initialization not implemented.')

  def tokenize(self, datasets):
    raise NotImplementedError('Tokenization not implemented.')

  @property
  def vocab_size(self):
    raise NotImplementedError('Vocabulary size not implemented.')

  @property
  def padding_token(self):
    return 0

  @property
  def max_token_seq_len(self):
    raise NotImplementedError('Maximum token sequence length not implemented.')

  def log(self, message, level='info'):
    return utils_log.log(
        message, self.config['accelerator'], self.logger, level=level
    )
