import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel

# coding=utf-8
# Copyright 2022 Microsoft Research and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch LayoutLMv3 model."""

import collections
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import apply_chunking_to_forward
from transformers.modeling_outputs import (
    BaseModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging


class LayoutLMv3MaskedImageModeling(nn.Module):
    def __init__(self,vocab_size=8192):
        super().__init__()
        self.model = AutoModel.from_pretrained("microsoft/layoutlmv3-base")
        self.lm_head = nn.Linear(768,vocab_size)

    def forward(self,
        input_ids=None,
        bbox=None,
        samples=None, # what is samples ? aka pixel_values
        bool_masked_pos=None,  # what is samples ?
        return_all_tokens=False
        ):
        out = self.model(
            input_ids=input_ids,
            bbox=bbox,
            pixel_values=samples,
            bool_masked_pos=bool_masked_pos
        )
        # TODO : insert bool_masked_pos into code 
            # bool_masked_pos=bool_masked_pos
        out = out.last_hidden_state
        out = out[:, 1:] # remove CLS token

        if return_all_tokens:
            out = self.lm_head(out)
            return out
        else:
            # return the masked tokens
            out = self.lm_head(out[bool_masked_pos])
            return out
