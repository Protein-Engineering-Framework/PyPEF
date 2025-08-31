# PyPEF - Pythonic Protein Engineering Framework
# https://github.com/niklases/PyPEF

# Some helper functions for infernece of different models 
# based on simple/wrapping functions

import numpy as np

from pypef.utils.helpers import get_device
from pypef.llm.utils import get_batches
from pypef.llm.esm_lora_tune import esm_setup, esm_tokenize_sequences, esm_infer
from pypef.llm.prosst_lora_tune import prosst_setup, prosst_tokenize_sequences, prosst_infer

import logging
logger = logging.getLogger('pypef.llm.inference')


def llm_embedder(llm_dict, seqs, verbose=True):
    try:
        np.shape(seqs)
    except ValueError:
        raise SystemError("Unequal input sequence length detected!")
    if list(llm_dict.keys())[0] == 'esm1v':
        x_llm_seqs, _attention_mask = esm_tokenize_sequences(
            seqs, tokenizer=llm_dict['esm1v']['llm_tokenizer'], 
            max_length=len(seqs[0]), verbose=verbose
        )
    elif list(llm_dict.keys())[0] == 'prosst':
        x_llm_seqs = prosst_tokenize_sequences(
            seqs, vocab=llm_dict['prosst']['llm_vocab'], verbose=verbose
        )
    else:
        raise SystemError(f"Unknown LLM dictionary input:\n{list(llm_dict.keys())[0]}")
    return x_llm_seqs


def inference(
        sequences,
        llm: str,
        pdb_file: str | None = None,
        wt_seq: str | None = None,
        device: str| None = None,
        model = None,
        verbose: bool = True
):
    """
    Inference of input or base model.
    """
    if device is None:
        device = get_device()
    if llm == 'esm':
        logger.info("Zero-shot LLM inference on test set using ESM1v...")
        llm_dict = esm_setup(sequences, verbose=verbose)
        if model is None:
            model = llm_dict['esm1v']['llm_base_model']
        x_llm_test = llm_embedder(llm_dict, sequences, verbose)
        y_test_pred = esm_infer(#llm_dict['esm1v']['llm_inference_function'](
            xs=get_batches(x_llm_test, batch_size=1, dtype=int), 
            attention_mask=llm_dict['esm1v']['llm_attention_mask'], 
            model=model, 
            device=device,
            verbose=verbose
        ).cpu()
    elif llm == 'prosst':
        logger.info("Zero-shot LLM inference on test set using ProSST...")
        llm_dict = prosst_setup(
            wt_seq, pdb_file, sequences=sequences, verbose=verbose
        )
        if model is None:
            model = llm_dict['prosst']['llm_base_model']
        x_llm_test = llm_embedder(llm_dict, sequences, verbose)
        y_test_pred = prosst_infer(#llm_dict['prosst']['llm_inference_function'](
            xs=x_llm_test, 
            model=model, 
            input_ids=llm_dict['prosst']['input_ids'], 
            attention_mask=llm_dict['prosst']['llm_attention_mask'], 
            structure_input_ids=llm_dict['prosst']['structure_input_ids'],
            verbose=verbose,
            device=device
        ).cpu()
    else:
        raise RuntimeError("Unknown LLM option.")
    return y_test_pred