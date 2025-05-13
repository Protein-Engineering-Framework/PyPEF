# Niklas Siedhoff
# PyPEF - Pythonic Protein Engineering Framework

# Using (training, testing/infering) ESM model(s) (e.g. ESM1v) published under 
# MIT License
# Copyright (c) Meta Platforms, Inc. and affiliates.
# https://github.com/facebookresearch/esm
# ESM1v model publication:
# Joshua Meier, Roshan Rao, Robert Verkuil, Jason Liu, Tom Sercu, Alexander Rives
# Language models enable zero-shot prediction of the effects of mutations on protein function
# bioRxiv 2021.07.09.450648; doi: https://doi.org/10.1101/2021.07.09.450648 

# Inspired by ConFit
# https://github.com/luo-group/ConFit


from __future__ import annotations

import logging
logger = logging.getLogger('pypef.llm.esm_lora_tune')

import torch
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm


from peft import LoraConfig, get_peft_model
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()
from transformers import EsmForMaskedLM, EsmTokenizer

from pypef.utils.helpers import get_device


def get_esm_models():
    base_model = EsmForMaskedLM.from_pretrained(f'facebook/esm1v_t33_650M_UR90S_3')
    tokenizer = EsmTokenizer.from_pretrained(f'facebook/esm1v_t33_650M_UR90S_3')
    peft_config = LoraConfig(r=8, target_modules=["query", "value"])
    lora_model = get_peft_model(base_model, peft_config)
    optimizer = torch.optim.Adam(lora_model.parameters(), lr=0.01)
    return base_model, lora_model, tokenizer, optimizer


def esm_tokenize_sequences(sequences, tokenizer, max_length, verbose=True):
    encoded_sequences = []
    for seq in tqdm(sequences, desc='Tokenizing sequences for ESM modeling', disable=not verbose):
        encoded_sequence, attention_mask = tokenizer(
            seq, 
            padding='max_length', 
            truncation=True, 
            max_length=max_length
        ).values()
        encoded_sequences.append(encoded_sequence)
    return encoded_sequences, attention_mask


def get_y_pred_scores(encoded_sequences, attention_masks, 
                      model, device: str | None = None):
    if device is None:
        device = get_device()
    model = model.to(device)
    out = model(encoded_sequences.to(device), attention_masks.to(device), 
                output_hidden_states=True)
    logits = out.logits
    token_probs = torch.log_softmax(logits, dim=-1)
    for i_s, sequence in enumerate(encoded_sequences):
        for i_aa, aa in enumerate(sequence):
            # alternative: use Tensor.index_select() function
            if i_aa == 0:
                seq_log_probs = token_probs[i_s, i_aa, aa].reshape(1)
            else:
                seq_log_probs = torch.cat(
                    (seq_log_probs, token_probs[i_s, i_aa, aa].reshape(1)), 0)
        if i_s == 0:
            log_probs = torch.sum(torch.Tensor(seq_log_probs)).reshape(1)
        else:
            log_probs = torch.cat(
                (log_probs, torch.sum(torch.Tensor(seq_log_probs)).reshape(1)), 0)
    return log_probs


def corr_loss(y_true: torch.Tensor, y_pred: torch.Tensor):
    res_true = y_true - torch.mean(y_true)
    res_pred = y_pred - torch.mean(y_pred)
    cov = torch.mean(res_true * res_pred)
    var_true = torch.mean(res_true**2)
    var_pred = torch.mean(res_pred**2)
    sigma_true = torch.sqrt(var_true)
    sigma_pred = torch.sqrt(var_pred)
    return - cov / (sigma_true * sigma_pred)


def get_batches(a, dtype, batch_size=5, 
                keep_numpy: bool = False, keep_remaining=False, verbose: bool = False):
    a = np.asarray(a, dtype=dtype)
    orig_shape = np.shape(a)
    remaining = len(a) % batch_size
    if remaining != 0:
        if len(a) > batch_size:
            a = a[:-remaining]
            a_remaining = a[-remaining:]
        else:
            logger.info(f"Batch size greater than or equal to total array length: "
                  f"returning full array (of shape: {np.shape(a)})...")
            if keep_remaining:
                return list(a)
            else:
                return a
    if len(orig_shape) == 2:
        a = a.reshape(np.shape(a)[0] // batch_size, batch_size, np.shape(a)[1])
    else:  # elif len(orig_shape) == 1:
        a = a.reshape(np.shape(a)[0] // batch_size, batch_size)
    new_shape = np.shape(a)
    if verbose:
        logger.info(f'{orig_shape} -> {new_shape}  (dropped {remaining})')
    if keep_remaining: # Returning a list
        a = list(a)
        logger.info('Adding dropped back to batches as last batch...')
        a.append(a_remaining)
        return a
    if keep_numpy:
        return a
    return torch.Tensor(a).to(dtype)
    

def esm_test(xs, attention_mask, scores, loss_fn, model, device: str | None = None):
    if device is None:
        device = get_device()
    attention_masks = torch.Tensor(np.full(
        shape=np.shape(xs), fill_value=attention_mask)).to(torch.int64)
    logger.info(f'Infering ESM model for testing using {device.upper()} device...')
    model = model.to(device)
    xs, attention_masks, scores = (
        torch.Tensor(xs).to(device), attention_masks.to(device), 
        torch.Tensor(scores).to(torch.float).to(device)
    )
    pbar_epochs = tqdm(zip(xs, attention_masks, scores), total=len(xs))
    for i ,(xs_b, attns_b, scores_b) in enumerate(pbar_epochs):
        xs_b, attns_b = xs_b.to(torch.int64), attns_b.to(torch.int64)
        with torch.no_grad():
            y_preds = get_y_pred_scores(xs_b, attns_b, model, device)
            if i == 0:
                y_preds_total = y_preds
                scores_total = scores_b
            else:
                y_preds_total = torch.cat((y_preds_total, y_preds))
                scores_total = torch.cat((scores_total, scores_b))
        batch_loss = loss_fn(scores_b, y_preds)
        total_loss = loss_fn(torch.flatten(scores_total), torch.flatten(y_preds_total))
        batch_scorr = spearmanr(scores_b.cpu(), y_preds.cpu())[0]
        total_scorr = spearmanr(scores_total.cpu(), y_preds_total.cpu())[0]
        pbar_epochs.set_description(
            f"Testing: Batch {i + 1}/{len(xs)} | Batch loss: {batch_loss:.4f} (SpearCorr: "
            f"{batch_scorr:.4f})| Total loss: {total_loss:.4f} (SpearCorr: {total_scorr:.4f})")
    logger.info(f"Test performance: Loss: {total_loss:.4f}, SpearCorr: {total_scorr:.4f}")
    return torch.flatten(scores).detach().cpu(), torch.flatten(y_preds_total).detach().cpu()


def esm_infer(xs, attention_mask, model, device: str | None = None, verbose=True):
    if device is None:
        device = get_device()
    attention_masks = torch.Tensor(np.full(
        shape=np.shape(xs), fill_value=attention_mask)).to(torch.int64)
    if verbose:
        logger.info(f'Infering ESM model for predictions using {device.upper()} device...')
    for i , (xs_b, am_b) in enumerate(tqdm(
        zip(xs, attention_masks), total=len(xs), 
        desc="ESM inference - processing sequences", disable=not verbose
    )):
        xs_b = xs_b.to(torch.int64)
        with torch.no_grad():
            y_preds = get_y_pred_scores(xs_b, am_b, model, device)
            if i == 0:
                y_preds_total = y_preds
            else:
                y_preds_total = torch.cat((y_preds_total, y_preds))
    return torch.flatten(y_preds_total)


def esm_train(xs, attention_mask, scores, loss_fn, model, optimizer, n_epochs=3, 
              device: str | None = None, seed: int | None = None):
    if seed is not None:
        torch.manual_seed(seed)
    if device is None:
        device = get_device()
    logger.info(f'Training ESM model using {device.upper()} device '
          f'(N_Train={len(torch.flatten(scores))})...')
    model = model.to(device)
    attention_masks = torch.Tensor(np.full(
        shape=np.shape(xs), fill_value=attention_mask)).to(torch.int64)
    xs, attention_masks, scores = xs.to(device), attention_masks.to(device), scores.to(device) 
    pbar_epochs = tqdm(range(1, n_epochs + 1))
    loss = np.nan
    for epoch in pbar_epochs:
        try:
            pbar_epochs.set_description(f'Epoch: {epoch}/{n_epochs}. Loss: {loss.detach():>1f}')
        except AttributeError:
            pbar_epochs.set_description(f'Epoch: {epoch}/{n_epochs}')
        model.train()
        pbar_batches = tqdm(zip(xs, attention_masks, scores), total=len(xs), leave=False)
        for batch, (xs_b, attns_b, scores_b) in enumerate(pbar_batches):
            xs_b, attns_b = xs_b.to(torch.int64), attns_b.to(torch.int64)
            y_preds_b = get_y_pred_scores(xs_b, attns_b, model, device=device)
            loss = loss_fn(scores_b, y_preds_b)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            pbar_batches.set_description(
                f"Epoch: {epoch}. Loss: {loss.detach():>1f}  "
                f"[batch: {batch+1}/{len(xs)} | "
                f"sequence: {(batch + 1) * len(xs_b):>5d}/{len(xs) * len(xs_b)}]  "
            )
    y_preds_b = y_preds_b.detach()
    model.train(False)


def esm_setup(sequences, device: str | None = None):
    esm_base_model, esm_lora_model, esm_tokenizer, esm_optimizer = get_esm_models()
    esm_base_model = esm_base_model.to(device)
    x_esm, esm_attention_mask = esm_tokenize_sequences(
        sequences, esm_tokenizer, max_length=len(sequences[0]))
    llm_dict_esm = {
        'esm1v': {
            'llm_base_model': esm_base_model,
            'llm_model': esm_lora_model,
            'llm_optimizer': esm_optimizer,
            'llm_train_function': esm_train,
            'llm_inference_function': esm_infer,
            'llm_loss_function': corr_loss,
            'x_llm' : x_esm,
            'llm_attention_mask':  esm_attention_mask,
            'llm_tokenizer': esm_tokenizer
        }
    }
    return llm_dict_esm
