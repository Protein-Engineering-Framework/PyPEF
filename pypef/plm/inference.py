# PyPEF - Pythonic Protein Engineering Framework
# https://github.com/niklases/PyPEF

# Some helper functions for infernece of different models 
# based on simple/wrapping functions

import os
import inspect
from functools import partial
from typing import Literal
import numpy as np
from scipy.stats import spearmanr
import torch
import torch.nn.functional as F
from Bio import SeqIO

from pypef.utils.helpers import tqdm
from pypef.plm.prosst_lora_tune import get_prosst_models, get_structure_quantizied
from pypef.utils.helpers import get_device
from pypef.plm.utils import extract_mean_or_pos_embeddings, hybrid_corr_mse_loss, get_batches
from pypef.plm.esm_lora_tune import get_esm_models


import logging
logger = logging.getLogger('pypef.plm.inference')


def checkpoint(model, filename):
    """
    Saves only the adapter weights if it's a PEFT model, 
    otherwise saves the full state dict.
    """
    #if isinstance(model, PeftModel):
    #    state_dict = get_peft_model_state_dict(model)
    #    torch.save(state_dict, filename)
    #else:
    torch.save(model.state_dict(), filename)


def load_model(model, filename):
    """
    Loads weights safely depending on model type.
    """
    logger.info(f'Loading model weights from: {os.path.abspath(filename)}')
    state_dict = torch.load(filename, weights_only=True)
    
    #if isinstance(model, PeftModel):
    #    set_peft_model_state_dict(model, state_dict)
    #else:
    model.load_state_dict(state_dict)


def tokenize_sequences(sequences, tokenizer, max_length=None, verbose=True):
    model_name = getattr(tokenizer, "name_or_path", None) or type(tokenizer).__name__
    if max_length is None:
        if verbose:
            logger.info(f"Setting max. tokenized sequence length to {len(sequences[0]) + 2}, tokenizer: {model_name}...")
        max_length = len(sequences[0]) + 2
    tokenized_sequences = []
    for seq in tqdm(sequences, desc='Tokenizing sequences', disable=not verbose):
        encoded_sequence, attention_mask = tokenizer(
            seq, 
            padding='max_length', 
            truncation=True,  # False for not uniform length distribution (truncation) 
            max_length=max_length
        ).values()
        tokenized_sequences.append(encoded_sequence)
    return tokenized_sequences, attention_mask


def sequence_log_likelihood( 
        attention_mask, 
        wt_input_ids,
        model, 
        tokenized_sequences = None, # Not needed for WT-only extraction
        scoring_mode: str = "wt-marginal",   # "wt-marginal" | "full-sequence"
        train: bool = False,
        cut_special_tokens: bool = True,  # assumption: cut first and last token
        device=None,
        verbose: bool = False,
        **model_kwargs
    ):
    """
    Unified scoring function.

    scoring_mode:
        - "wt-marginal": forward pass on WT only (fast, approximate)
        - "full-sequence": forward pass per sequence (exact PLL)

    Returns:
        torch.Tensor of shape [num_sequences]
    """
    extract_emb = model_kwargs.pop("extract_emb", False)
    extract_probs = model_kwargs.pop("extract_probs", False) # Extract raw probabilities
    tokenizer = model_kwargs.pop("tokenizer", None)          # Required if extract_probs=True
    assert scoring_mode in ["wt-marginal", "full-sequence"]
    if device is None:
        device = get_device()
    if wt_input_ids.dim() == 1:
        wt_input_ids = wt_input_ids.unsqueeze(0)
    wt_input_ids = wt_input_ids.to(device)
    log_probs = []
    #structure_input_ids = model_kwargs.get("wt_structure_input_ids", None)
    if scoring_mode == "wt-marginal":
        attention_masks = torch.Tensor(np.full(
            shape=np.shape(wt_input_ids), fill_value=attention_mask)).to(torch.int64).to(device)
        try:
            if train:
                outputs = model(
                    input_ids=wt_input_ids,
                    attention_mask=attention_masks,
                    output_hidden_states=False,
                    return_dict=True,
                    **model_kwargs
                )

            else:
                with torch.no_grad():
                    outputs = model(
                        input_ids=wt_input_ids,
                        attention_mask=attention_masks,
                        output_hidden_states=False,
                        return_dict=True,
                        **model_kwargs
                    )
        except TypeError as e:
            logger.info(f"Did not find model input keyword arguments (kwargs: "
                  f"{model_kwargs.keys()}). Available kawrgs identified from "
                  f"model.forward function inspect:\n"
                  f"{inspect.signature(model.forward)}\nOriginal error:")
            raise e


        logits = outputs.logits
        logits = logits.squeeze(0)   # remove batch dim
        # Make sure that special tokens are always removed / masked 
        # and only pure amino acid sequence tokens are present / unmasked
        if tokenized_sequences is not None:
            tokenized_seq_len = tokenized_sequences.shape[1]
        else:
            tokenized_seq_len = wt_input_ids.shape[1]
        if cut_special_tokens:
            logits = logits[1:-1]        # drop CLS/EOS
            tokenized_seq_len -= 2
        if extract_probs:
            if tokenizer is None:
                raise RuntimeError(
                    "For getting the conditional amino acid probability, "
                    "a tokenizer has to be provided to convert tokens into IDs."
                )
            aa_conditional_prob_order = model_kwargs.pop("kermut_aa_order", [
                "A", "C", "D", "E", "F", "G", "H", "I", "K", "L", 
                "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"
            ])
            
            # Map ProSST's internal vocab indices to this explicit layout
            aa_token_ids = []
            for aa in aa_conditional_prob_order:
                token_id = tokenizer.convert_tokens_to_ids(aa)
                if token_id == tokenizer.unk_token_id or token_id is None:
                    raise ValueError(f"ProSST Tokenizer failed to locate vocab ID for amino acid '{aa}'.")
                aa_token_ids.append(token_id)
            
            # Filter the raw logits down to these exact 20 matching columns
            filtered_logits = logits[:, aa_token_ids]  # Shape: [seq_len, 20]
            
            # Convert to raw probability space to match 'p_mean = np.exp(log_p_mean)'
            conditional_probs = F.softmax(filtered_logits, dim=-1) 
            return conditional_probs
        
        token_probs = torch.log_softmax(logits, dim=-1)
        assert tokenized_seq_len == token_probs.shape[0], (
            f"{tokenized_seq_len} != {token_probs.shape[0]}")

        for tokenized_seq in tokenized_sequences:
            if cut_special_tokens:
                tokenized_seq = tokenized_seq[1:-1]

            seq_lp = token_probs[
                torch.arange(tokenized_seq.shape[0], device=tokenized_seq.device),
                tokenized_seq
            ].sum(dtype=torch.float64)

            log_probs.append(seq_lp)

    elif scoring_mode == "full-sequence":
        if extract_emb:
            embeddings = []
        for tokenized_seq in tokenized_sequences:
            if tokenized_seq.dim() == 1:
                tokenized_seq = tokenized_seq.unsqueeze(0)

            attention_masks = torch.Tensor(np.full(
                shape=tokenized_seq.shape,
                fill_value=attention_mask)
            ).to(torch.int64).to(device)

            try:
                if train:
                    outputs = model(
                        input_ids=tokenized_seq,
                        attention_mask=attention_masks,
                        return_dict=True,
                        **model_kwargs
                    )
                else:
                    with torch.no_grad():
                        outputs = model(
                            input_ids=tokenized_seq,
                            attention_mask=attention_masks,
                            return_dict=True,
                            output_hidden_states=extract_emb,
                            **model_kwargs
                        )
                    if extract_emb:
                        # extractinf full embeddings here, so batching is 
                        # required for less memory consumption
                        # Only returning last hidden layer!
                        token_embeddings = outputs.hidden_states[-1]  # (1, L+2, D)
                        # exclude CLS/EOS
                        seq_embedding = token_embeddings[0, 1:-1] # (L, D)
                        embeddings.append(seq_embedding)
                        continue

            except TypeError as e:
                logger.info(f"Did not find model input keyword arguments (kwargs: "
                    f"{model_kwargs.keys()}). Available kawrgs identified from "
                    f"model.forward function inspect:\n"
                    f"{inspect.signature(model.forward)}\nOriginal error:"
                )
                raise e

            logits = outputs.logits.squeeze(0)

            if cut_special_tokens:
                logits = logits[1:-1]
                target_tokens = tokenized_seq.squeeze(0)[1:-1]
            else:
                target_tokens = tokenized_seq.squeeze(0)

            token_log_probs = torch.log_softmax(logits, dim=-1)

            seq_lp = token_log_probs[
                torch.arange(target_tokens.shape[0], device=device),
                target_tokens
            ].sum(dtype=torch.float64)

            log_probs.append(seq_lp)
    if extract_emb:
        return torch.stack(embeddings)
    return torch.stack(log_probs)


def mutation_only_mutation_masked_pll(
    tokenized_sequences: torch.Tensor,        # (L,)
    wt_input_ids: torch.Tensor,               # (L,)
    attention_mask: torch.Tensor,             # (L,)
    model,
    mask_token_id: int,
    train: bool = False,
    device: str | None = None,
    verbose: bool = False,
    **kwargs
):
    """
    Correct mutation-only pseudo-log-likelihood for sequences.
    """
    tokenized_sequences = tokenized_sequences.to(device)
    structure_input_ids = kwargs.get("ss_input_ids", None)
    if structure_input_ids is not None:
        assert structure_input_ids.shape[1] == tokenized_sequences.shape[1], (
            f"{structure_input_ids.shape[1]} != {tokenized_sequences.shape[1]}")
        structure_input_ids = structure_input_ids.to(device)
    if wt_input_ids.dim() == 2 and wt_input_ids.shape[0] == 1:
        wt_input_ids = wt_input_ids.squeeze(0)
    wt_input_ids = wt_input_ids.to(device)
    if attention_mask.dim() == 2 and attention_mask.shape[0] == 1:
        attention_mask = attention_mask.squeeze(0)
    attention_mask = attention_mask.to(device)
    plls = torch.empty(len(tokenized_sequences), device=device)
    for i, tokenized_seq in enumerate(tokenized_sequences):
        assert tokenized_seq.dim() == 1
        assert wt_input_ids.dim() == 1
        assert attention_mask.dim() == 1
        assert tokenized_seq.shape == wt_input_ids.shape == attention_mask.shape
        pll = torch.tensor(0.0, device=device)

        # Identify mutated positions (exclude padding, CLS, EOS)
        diff = (tokenized_seq != wt_input_ids) & (attention_mask == 1)
        diff[0] = False
        diff[-1] = False

        mutated_positions = diff.nonzero(as_tuple=False).flatten()
        # n_mutations = (tokenized_seq != wt_input_ids).sum().item()
        # Mutated positions: [int(m) - 1 for m in mutated_positions.cpu()]  # Remove CLS token position

        for pos in tqdm(
            mutated_positions,
            desc="Masked PLL (single sequence)",
            disable=not verbose
        ):
            masked_input_ids = tokenized_seq.clone()
            masked_input_ids[pos] = mask_token_id
            if structure_input_ids is not None:
                masked_ss_input_ids = structure_input_ids.clone()
                masked_ss_input_ids[0, pos] = mask_token_id

            if train:
                if structure_input_ids is not None:
                    outputs = model(
                        input_ids=masked_input_ids.unsqueeze(0),
                        attention_mask=attention_mask.unsqueeze(0),
                        ss_input_ids=masked_ss_input_ids  # Check
                    )
                else:
                    outputs = model(
                        input_ids=masked_input_ids.unsqueeze(0),
                        attention_mask=attention_mask.unsqueeze(0),
                        output_hidden_states=False
                    )
            else:
                with torch.no_grad():
                    if structure_input_ids is not None:
                        outputs = model(
                            input_ids=masked_input_ids.unsqueeze(0),
                            attention_mask=attention_mask.unsqueeze(0),
                            ss_input_ids=masked_ss_input_ids  # Check
                        )
                    else:
                        outputs = model(
                            input_ids=masked_input_ids.unsqueeze(0),
                            attention_mask=attention_mask.unsqueeze(0),
                            output_hidden_states=False
                        )
            logits = outputs.logits  # (1, L, V)
            log_probs = F.log_softmax(logits[0, pos], dim=-1)
            true_token = tokenized_seq[pos]
            pll = pll + log_probs[true_token]
        
        plls[i] = pll

    return plls


def mutation_all_pos_masked_pll(
    tokenized_sequences: torch.Tensor,    # (L,)
    attention_mask: torch.Tensor,         # (L,)
    model,
    mask_token_id: int,
    train: bool = False,
    device: str | None = None,
    verbose: bool = False,
    **kwargs
):
    """
    Correct mutation-only pseudo-log-likelihood for sequences.
    """
    structure_input_ids = kwargs.get("ss_input_ids", None)
    if structure_input_ids is not None:
        assert structure_input_ids.shape[1] == tokenized_sequences.shape[1], (
            f"{structure_input_ids.shape[1]} != {tokenized_sequences.shape[1]}")
        structure_input_ids = structure_input_ids.to(device)
    tokenized_sequences = tokenized_sequences.to(device)
    if attention_mask.dim() == 2 and attention_mask.shape[0] == 1:
        attention_mask = attention_mask.squeeze(0)
    attention_mask = attention_mask.to(device)
    plls = torch.empty(len(tokenized_sequences), device=device)
    for i, tokenized_seq in enumerate(tokenized_sequences):
        L = tokenized_seq.shape[0]
        pll = torch.tensor(0.0, device=device)

        # Positions to score: all real tokens except CLS/EOS
        positions = (attention_mask == 1).nonzero(as_tuple=False).flatten()
        positions = positions[(positions != 0) & (positions != L - 1)]


        for pos in tqdm(
            positions,
            desc="Masked PLL (single sequence)",
            disable=not verbose
        ):
            masked_input_ids = tokenized_seq.clone()
            masked_input_ids[pos] = mask_token_id

            if structure_input_ids is not None:
                masked_ss_input_ids = structure_input_ids.clone()
                masked_ss_input_ids[0, pos] = mask_token_id

            if train:
                if structure_input_ids is not None:
                    outputs = model(
                        input_ids=masked_input_ids.unsqueeze(0),
                        attention_mask=attention_mask.unsqueeze(0),
                        ss_input_ids=masked_ss_input_ids  # Check 
                    )
                else:
                    outputs = model(
                        input_ids=masked_input_ids.unsqueeze(0), 
                        attention_mask=attention_mask.unsqueeze(0), 
                        output_hidden_states=False
                    )
            else:
                with torch.no_grad():
                    if structure_input_ids is not None:
                        outputs = model(
                                input_ids=masked_input_ids.unsqueeze(0),
                                attention_mask=attention_mask.unsqueeze(0),
                                ss_input_ids=masked_ss_input_ids  # Check
                        )
                    else:
                        outputs = model(
                            input_ids=masked_input_ids.unsqueeze(0), 
                            attention_mask=attention_mask.unsqueeze(0), 
                            output_hidden_states=False
                        )
            logits = outputs.logits  # (1, L, V)

            log_probs = F.log_softmax(logits[0, pos], dim=-1)
            true_token = tokenized_seq[pos]
            pll = pll + log_probs[true_token]

        plls[i] = pll

    return plls


def plm_inference(
    tokenized_sequences,
    wt_input_ids,
    attention_mask,
    model,
    mask_token_id = None,
    inference_type='wt-marginal-log-likelihood',
    extract_emb: bool = False,
    extract_conditional_aa_prob: bool = False,
    batch_size: int | None = 5,
    train=False,
    device=None,
    verbose=False,
    **kwargs
):  
    if device is None:
        device = get_device()
    model = model.to(device)

    if train:
        model.train()
        keep_remaining = False
    else:
        model.eval()
        keep_remaining = True

    model_name = getattr(model, "name_or_path", None) or type(model).__name__
    model = model.to(device)
    model_kwargs = {}

    if not isinstance(attention_mask, torch.Tensor):
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    scoring_mode = None
    if inference_type == 'mutation-masking':
        inference_function = mutation_only_mutation_masked_pll
    elif inference_type in ['full-masking', 'all-pos-masking']:
        inference_function = mutation_all_pos_masked_pll
    # Unmasked
    elif inference_type in ['wt-marginal', 'wt-marginal-log-likelihood']:
        inference_function = sequence_log_likelihood
        scoring_mode = "wt-marginal"
    elif inference_type in ['full-sequence', 'full-sequence-log-likelihood']:
        inference_function = sequence_log_likelihood
        scoring_mode = "full-sequence"
    else:
        raise SystemError(
            f"Choose between 'wt-marginal-log-likelihood', "
            f"'full-sequence-log-likelihood', 'mutation-masking', "
            f"and 'full-masking', got {inference_type}.")
    if extract_emb:
        if verbose:
            logger.info(f"Extracting sequence embeddings using the 'full-sequence-log-likelihood' "
                        f"function")
        inference_function = sequence_log_likelihood
        scoring_mode = "full-sequence"
        model_kwargs["extract_emb"] = True
        model_kwargs["extract_probs"] = False
    elif extract_conditional_aa_prob:
        if verbose:
            logger.info(f"Extracting conditional amino acid probabilites using the 'wt-marginal-"
                        f"sequence-log-likelihood' function")
        inference_function = sequence_log_likelihood
        scoring_mode = "wt-marginal"
        model_kwargs["extract_emb"] = False
        model_kwargs["extract_probs"] = True
        model_kwargs["tokenizer"] = kwargs["tokenizer"]

    scores = []
    if batch_size is None and tokenized_sequences is not None:
        xs_b = torch.atleast_2d(tokenized_sequences)
    else:
        if verbose:
            logger.info(f"Splitting tokenized sequences into batches...")
        if extract_conditional_aa_prob and tokenized_sequences is None:
            xs_b = [None]
        else:
            xs_b = get_batches(tokenized_sequences, dtype=int, batch_size=batch_size,
                               keep_remaining=keep_remaining, verbose=verbose)
            xs_b = [torch.from_numpy(x).to(device) for x in xs_b]
    if extract_emb:
        desc = (f"PLM {model_name} inference: embeddings batch "
                f"(size={batch_size}) processing ({device.upper()})")
    elif extract_conditional_aa_prob:
        desc = (f"PLM {model_name} inference: AA probabilities batch (size={batch_size}) "
                f"processing ({device.upper()})'")
    else:
        desc = (f"PLM {model_name} inference: {inference_type} batch (size={batch_size}) "
                f"processing ({device.upper()})'")

    if mask_token_id is not None:
        model_kwargs["mask_token_id"] = mask_token_id

    wt_structure_input_ids = kwargs.get('wt_structure_input_ids')
    if wt_structure_input_ids is not None:
        model_kwargs["ss_input_ids"] = wt_structure_input_ids.to(device)

    pbar = tqdm(
        xs_b,
        desc=desc,
        disable=not verbose
    )

    with torch.set_grad_enabled(train):
        for x in pbar:
            pll = inference_function(
                attention_mask=attention_mask,
                wt_input_ids=wt_input_ids,
                tokenized_sequences=x,
                model=model,
                train=train,
                scoring_mode=scoring_mode,
                device=device,
                verbose=False,
                **model_kwargs
            )
            scores.append(pll)
                    
    return torch.cat(scores)


def plm_train(
        x_sequences, 
        scores, 
        loss_fn, 
        model, 
        optimizer,
        wt_input_ids, 
        attention_mask, 
        batch_size: int = 5,
        n_epochs=50, 
        device: str | None = None, 
        seed: int | None = None,
        early_stop: int = 50, 
        verbose: bool = True, 
        n_batch_grad_accumulations: int = 1, 
        raise_error_on_train_fail: bool = True,
        progress_cb=None, 
        abort_cb=None,
        **kwargs
):
    """
    Wrapper function for `plm_inference()` for PLM training.
    """
    if seed is not None:
        torch.manual_seed(seed)
    if device is None:
        device = get_device()
    wt_structure_input_ids = kwargs.get('wt_structure_input_ids')
    logger.info(f"Model training using {device.upper()} device "
          f"(N_Train={len(scores)})...")
    scores_batched = torch.from_numpy(
        get_batches(scores, dtype=float, batch_size=batch_size,
                    keep_remaining=False, verbose=True)
    )
    x_sequences_batched = torch.from_numpy(
        get_batches(x_sequences, dtype=int, batch_size=batch_size, 
                    keep_remaining=False, verbose=True)
    )
    x_sequences_batched = x_sequences_batched.to(device)
    scores_batched = scores_batched.to(device)
    pbar_epochs = tqdm(range(1, n_epochs + 1), disable=not verbose)
    epoch_spearman_1 = -1.0
    did_not_improve_counter = 0
    best_model = None
    best_model_epoch = np.nan
    best_model_perf = np.nan
    loss = np.nan
    os.makedirs('model_saves', exist_ok=True)
    for epoch in pbar_epochs:
        if epoch == 0:
            pbar_epochs.set_description(f'Epoch {epoch}/{n_epochs}')
        model.train()
        y_preds_detached = []
        pbar_batches = tqdm(
            zip(x_sequences_batched, scores_batched),
            total=len(x_sequences_batched), leave=False, disable=not verbose
        )
        for batch, (seqs_b, scores_b) in enumerate(pbar_batches):
            if abort_cb and abort_cb():
                return
            if seqs_b.dim() == 2:
                seqs_b = seqs_b.unsqueeze(0)  # e.g., (5, 400)  -> (1, 5 400)
            y_preds_b = plm_inference(
                tokenized_sequences=seqs_b, 
                wt_input_ids=wt_input_ids, 
                attention_mask=attention_mask,
                model=model, 
                train=True, 
                batch_size=None, 
                device=device,
                verbose=False,
                wt_structure_input_ids=wt_structure_input_ids,                 
            )
            y_preds_detached.append(y_preds_b.detach().cpu().numpy().flatten())
            loss = loss_fn(scores_b, y_preds_b) / n_batch_grad_accumulations
            if progress_cb:
                progress_cb(epoch - 1, batch + 1, len(pbar_epochs), len(pbar_batches), loss)
            loss.backward()
            if (batch + 1) % n_batch_grad_accumulations == 0 or (batch + 1) == len(pbar_batches):
                optimizer.step()
                optimizer.zero_grad()
            pbar_batches.set_description(
                f"Epoch: {epoch}. Loss: {loss.detach():>1f} "
                f"[batch: {batch + 1}/{len(x_sequences_batched)} | "
                f"sequence: {(batch + 1) * len(x_sequences_batched[0]):>5d}/"
                f"{len(x_sequences)}] ({device.upper()})"
            )
        epoch_spearman_2 = spearmanr(scores_batched.cpu().numpy().flatten(),
                                     np.array(y_preds_detached).flatten())[0]
        if epoch_spearman_2 == np.nan:
            raise SystemError(
                f"No correlation between Y_true and Y_pred could be computed...\n"
                f"Y_true: {scores_batched.cpu().numpy().flatten()}, "
                f"Y_pred: {np.array(y_preds_detached)}"
            )
        if epoch_spearman_2 > epoch_spearman_1 or epoch == 0:
            if best_model is not None:
                if os.path.isfile(best_model):
                    os.remove(best_model)
            did_not_improve_counter = 0
            best_model_epoch = epoch
            best_model_perf = epoch_spearman_2
            best_model = (
                f"model_saves/Epoch{epoch}-Ntrain{len(scores_batched.cpu().numpy().flatten())}"
                f"-SpearCorr{epoch_spearman_2:.3f}.pt"
            )
            checkpoint(model, best_model)
            epoch_spearman_1 = epoch_spearman_2
            logger.info(f"Saved current best model as {best_model}")
        else:
            did_not_improve_counter += 1
            if did_not_improve_counter >= early_stop:
                logger.info(f'\nEarly stop at epoch {epoch}...')
                break
        loss_total = loss_fn(
            torch.flatten(scores_batched).to('cpu'),
            torch.flatten(torch.Tensor(np.array(y_preds_detached).flatten()))
        )
        pbar_epochs.set_description(
            f'Epoch {epoch}/{n_epochs} [SpearCorr: {epoch_spearman_2:.3f}, Loss: {loss_total:.3f}] '
            f'(Best epoch: {best_model_epoch}: {best_model_perf:.3f}) ({device.upper()})')
    if progress_cb:
        progress_cb(epoch, batch + 1, len(pbar_epochs), len(pbar_batches), loss)
    if best_model is None:
        msg = ("Failed to train a model (probably due to the input "
               "data characteristics and loss/correlation being NaN).")
        if raise_error_on_train_fail:
            raise RuntimeError(msg)
        else:
            logger.warning(f"{msg} Continuing nonetheless (using failed model "
                           f"and replacing NaN's with zeros)...")
            y_preds_train = plm_inference(
                x_sequences,
                wt_input_ids, 
                attention_mask, 
                model,
                wt_structure_input_ids=wt_structure_input_ids,
                train=False, 
                verbose=False
            )
            y_preds_train[torch.isnan(y_preds_train)] = 0.0
    else:        
        logger.info(f"Loading best model as {best_model}...")
        load_model(model, best_model)
        y_preds_train = plm_inference(
                x_sequences,
                wt_input_ids, 
                attention_mask, 
                model,
                wt_structure_input_ids=wt_structure_input_ids,
                train=False, 
                verbose=False
            )
    return y_preds_train.cpu()


def get_plm_embeddings(
        tokenized_sequences, 
        model,
        wt_input_ids,  # wt seq. token
        attention_mask,
        mode: Literal["mean", "positional"] = "mean", 
        extract_conditional_aa_prob: bool = False,
        batch_size: int = 100,
        variants: str | None = None,
        plm_inference_function=None,
        verbose: bool = True,
        device: str | None = None,
        **embedding_func_kwargs
):
    if device is None:
        device = get_device()
    model_name = getattr(model, "name_or_path", None) or type(model).__name__
    desc=f"Getting PLM {model_name} embeddings (mode={mode})"
    if extract_conditional_aa_prob:
        desc=f"Getting AA cond. probs. from PLM {model_name} embeddings"
    if plm_inference_function is None:
        plm_inference_function = plm_inference
    pbar = tqdm(range(0, len(tokenized_sequences), batch_size), desc=desc, disable=not verbose)
    extract_emb = True
    processed_embs = []
    for i in pbar:
        start_idx = i
        end_idx = min(i + batch_size, len(tokenized_sequences))
        batch_seqs = tokenized_sequences[start_idx:end_idx]
        if extract_conditional_aa_prob:
            extract_emb = False

        full_embs = plm_inference_function(
            tokenized_sequences=batch_seqs, model=model, wt_input_ids=wt_input_ids, 
            attention_mask=attention_mask, extract_emb=extract_emb, 
            extract_conditional_aa_prob=extract_conditional_aa_prob, 
            device=device, **embedding_func_kwargs
        )
        batch_variants = None
        if variants is not None:
            batch_variants = variants[start_idx:end_idx]
        if extract_conditional_aa_prob:
            embs = full_embs
        else:
            embs = extract_mean_or_pos_embeddings(full_embs, mode=mode, mutation_strings=batch_variants)
            
        pbar.set_description(f"{desc}: {tuple(full_embs.shape)}->{tuple(embs.shape)} "
                             f"({str(full_embs.device).upper().split(':')[0]})")
        processed_embs.append(embs)
        if end_idx >= len(tokenized_sequences):
            final_rows = sum(x.shape[0] for x in processed_embs)
            final_shape = (final_rows, *processed_embs[0].shape[1:])
            pbar.set_description(
                f"{desc}: final shape={final_shape} "
                f"({str(full_embs.device).upper().split(':')[0]})"
            )
    processed_embs = torch.cat(processed_embs, dim=0)
    return processed_embs


def esm_setup(
        wt_seq, 
        sequences, 
        model: str = "facebook/esm2_t33_650M_UR50D",
        loss_method: str = "spearman",
        seed: int | None =None,
        revision: str | None = None,
        device: str | None = None, 
        verbose: bool = True
):
    if device is None:
        device = get_device()
    allowed_methods = [
        "spearman", "pearson", "listMLE", "pairwise-margin",
        "spearman-hybrid", "pearson-hybrid", "listMLE-hybrid", 
        "pairwise-margin-hybrid"
    ]
    if loss_method not in allowed_methods:
        raise RuntimeError(f"Loss function must be within {allowed_methods}.")
    esm_base_model, esm_lora_model, esm_tokenizer, esm_optimizer = get_esm_models(model=model, seed=seed, revision=revision)
    esm_base_model.eval()
    esm_lora_model.eval()
    esm_base_model, esm_lora_model = esm_base_model.to(device), esm_lora_model.to(device)
    wt_tokens, _ = tokenize_sequences(
            [wt_seq],
            esm_tokenizer,
            max_length=len(wt_seq) + 2
    )
    x_esm, esm_attention_mask = tokenize_sequences(
        sequences, esm_tokenizer, max_length=len(wt_seq) + 2, verbose=verbose)
    llm_dict_esm = {
        'esm': {
            'llm_base_model': esm_base_model,
            'llm_model': esm_lora_model,
            'llm_optimizer': esm_optimizer,
            'llm_train_function': plm_train,
            'llm_inference_function': plm_inference,
            'llm_loss_function': partial(hybrid_corr_mse_loss, method=loss_method),
            'x_llm' : torch.tensor(x_esm),                            # TODO: Check: Not needed here?
            'llm_attention_mask':  torch.tensor(esm_attention_mask),  # "
            'wt_input_ids': torch.tensor(wt_tokens),                  # "
            'wt_structure_input_ids': None,
            'llm_tokenizer': esm_tokenizer
        }
    }
    return llm_dict_esm


def prosst_setup(
        wt_seq, 
        pdb_file, 
        sequences, 
        loss_method: str = "spearman",
        seed: int | None =None,
        revision: str | None = None,
        device: str | None = None, 
        verbose: bool = True
):
    if device is None:
        device = get_device()
    if wt_seq is None:
        raise RuntimeError(
            "Running ProSST requires a wild-type sequence "
            "FASTA file input for embedding sequences! "
            "Specify a FASTA file with the --wt flag."
        )
    if pdb_file is None:
        raise RuntimeError(
            "Running ProSST requires a PDB file input "
            "for embedding sequences! Specify a PDB file "
            "with the --pdb flag."
        )
    
    allowed_methods = [
        "spearman", "pearson", "listMLE", "pairwise-margin",
        "spearman-hybrid", "pearson-hybrid", "listMLE-hybrid", "pairwise-margin-hybrid"
    ]
    if loss_method not in allowed_methods:
        raise RuntimeError(f"Loss function must be within {allowed_methods}.")

    pdb_seq = str(list(SeqIO.parse(pdb_file, "pdb-atom"))[0].seq)
    if wt_seq != pdb_seq:
        logger.warning(
            f"Wild-type sequence is not matching PDB-extracted sequence:"
            f"\nWT sequence:\n{wt_seq}\nPDB sequence:\n{pdb_seq}"
        )
    prosst_base_model, prosst_lora_model, prosst_tokenizer, prosst_optimizer = get_prosst_models(seed=seed, revision=revision)
    prosst_base_model.eval()
    prosst_lora_model.eval()
    prosst_vocab = prosst_tokenizer.get_vocab()
    prosst_base_model, prosst_lora_model = prosst_base_model.to(device), prosst_lora_model.to(device)
    prosst_optimizer = torch.optim.Adam(prosst_lora_model.parameters(), lr=0.0001)
    input_ids, prosst_attention_mask, structure_input_ids = get_structure_quantizied(
        pdb_file, prosst_tokenizer, wt_seq, device=device, verbose=verbose
    )
    x_llm_train_prosst, _attention_mask = tokenize_sequences(
        sequences=sequences, tokenizer=prosst_tokenizer, 
        max_length=len(wt_seq) + 2, verbose=verbose
    )
    llm_dict_prosst = {
        'prosst': {
            'llm_base_model': prosst_base_model,
            'llm_model': prosst_lora_model,
            'llm_optimizer': prosst_optimizer,
            'llm_train_function': plm_train,
            'llm_inference_function': plm_inference,
            'llm_loss_function': partial(hybrid_corr_mse_loss, method=loss_method),
            'x_llm': x_llm_train_prosst,
            'llm_attention_mask': prosst_attention_mask,
            'llm_vocab': prosst_vocab,
            'wt_input_ids': input_ids,
            'wt_structure_input_ids': structure_input_ids,
            'llm_tokenizer': prosst_tokenizer
        }
    }
    return llm_dict_prosst


class KNNFitnessRetrieval:
    def __init__(self, k=5):
        self.k = k
        self.train_embeddings = None
        self.train_labels = None

    def fit(self, train_embeddings, train_labels):
        """Stores the training variants' local profiles and fitness scores."""
        # Ensure tensors are on CPU/GPU consistently
        self.train_embeddings = F.normalize(train_embeddings, p=2, dim=-1)
        self.train_labels = train_labels.view(-1, 1)

    def retrieve(self, query_embeddings):
        """Finds the k-nearest training environments and aggregates their scores."""
        norm_queries = F.normalize(query_embeddings, p=2, dim=-1)
        
        # Compute Cosine Similarity Matrix: [N_query, N_train]
        similarity_matrix = torch.matmul(norm_queries, self.train_embeddings.T)
        
        # Get top-k nearest neighbors in the training set
        topk_sims, topk_indices = torch.topk(similarity_matrix, k=self.k, dim=-1)
        
        # Retrieve their corresponding true fitness scores
        # Shape: [N_query, k]
        retrieved_scores = self.train_labels[topk_indices].squeeze(-1)
        
        # Softmax weights based on similarities for a weighted average
        weights = F.softmax(topk_sims, dim=-1)
        
        # Compute the weighted features
        knn_mean_score = torch.sum(weights * retrieved_scores, dim=-1, keepdim=True)
        knn_max_score = torch.max(retrieved_scores, dim=-1, keepdim=True)[0]
        knn_min_score = torch.min(retrieved_scores, dim=-1, keepdim=True)[0]
        
        # Combine into a descriptive retrieval context vector
        return torch.cat([knn_mean_score, knn_max_score, knn_min_score], dim=-1)
    