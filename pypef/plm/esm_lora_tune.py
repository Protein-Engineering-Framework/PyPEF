# PyPEF - Pythonic Protein Engineering Framework
# https://github.com/niklases/PyPEF

# Using (training, testing/infering) ESM model(s) (e.g. ESM1v and ESM2) published under
# MIT License
# Copyright (c) Meta Platforms, Inc. and affiliates.
# https://github.com/facebookresearch/esm
# ESM1v model publication:
# Joshua Meier, Roshan Rao, Robert Verkuil, Jason Liu, Tom Sercu, Alexander Rives
# Language models enable zero-shot prediction of the effects of mutations on protein function
# bioRxiv 2021.07.09.450648; doi: https://doi.org/10.1101/2021.07.09.450648 
# ESM2 model publication:
# Zeming Lin, Halil Akin, Roshan Rao, Brian Hie, Zhongkai Zhu, Wenting Lu, Nikita Smetanin,
# Robert Verkuil, Ori Kabeli, Yaniv Shmueli, Allan dos Santos Costa, Maryam Fazel-Zarandi,
# Tom Sercu, Salvatore Candido, Alexander Rives
# Evolutionary-scale prediction of atomic-level protein structure with a language model
# Science 379, 1123-1130 (2023); doi: https://doi.org/10.1126/science.ade2574

# Inspired by ConFit
# https://github.com/luo-group/ConFit


from __future__ import annotations

import logging
logger = logging.getLogger('pypef.plm.esm_lora_tune')

import copy
import torch
from peft import LoraConfig, get_peft_model
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

from pypef.plm.utils import load_model_and_tokenizer
from pypef.plm.utils import _set_seeds


def get_esm_models(
        model='facebook/esm2_t33_650M_UR50D',
        seed: None | bool = None,
        revision: str | None = None,
        deepcopy_base_model: bool = True
):
    if seed is not None:
        _set_seeds(seed)
    base_model, tokenizer = load_model_and_tokenizer(
        model,
        revision=revision
        # Just sticking to AutoModelForMaskedLM and AutoTokenizer 
        # instead to EsmForMaskedLM and EsmTokenizer
    )
    for param in base_model.parameters():
        param.requires_grad = False
    base_model.eval()
    lora_base_model = copy.deepcopy(base_model) if deepcopy_base_model else base_model
    peft_config = LoraConfig(r=8, target_modules=["query", "value"])
    lora_model = get_peft_model(lora_base_model, peft_config)
    optimizer = torch.optim.Adam(lora_model.parameters(), lr=0.01)
    return base_model, lora_model, tokenizer, optimizer
