from dataclasses import dataclass
from transformers.modeling_outputs import MaskedLMOutput
from typing import Optional, Tuple
import torch


@dataclass
class KLMForReplacementAndMaskedLMOutput(MaskedLMOutput):
    replacement_logits: torch.FloatTensor = None
    replacement_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    replacement_attentions: Optional[Tuple[torch.FloatTensor]] = None
