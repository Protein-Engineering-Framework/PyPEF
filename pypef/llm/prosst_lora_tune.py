# Niklas Siedhoff
# PyPEF - Pythonic Protein Engineering Framework

# Using (training, testing/infering) ProSST model(s) published under 
# GNU GENERAL PUBLIC LICENSE: GPL-3.0 license
# Code repository: https://github.com/ai4protein/ProSST
# Mingchen Li, Pan Tan, Xinzhu Ma, Bozitao Zhong, Huiqun Yu, Ziyi Zhou, Wanli Ouyang, Bingxin Zhou, Liang Hong, Yang Tan
# ProSST: Protein Language Modeling with Quantized Structure and Disentangled Attention
# bioRxiv 2024.04.15.589672; doi: https://doi.org/10.1101/2024.04.15.589672 

import logging
logger = logging.getLogger('pypef.llm.prosst_lora_tune')

import os
import warnings

import torch
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from Bio import SeqIO, BiopythonParserWarning
warnings.filterwarnings(action='ignore', category=BiopythonParserWarning)

from pypef.llm.esm_lora_tune import corr_loss
from pypef.llm.prosst_structure.quantizer import PdbQuantizer
from pypef.utils.helpers import get_device


def prosst_tokenize_sequences(sequences, vocab, verbose=True):
    sequences = np.atleast_1d(sequences).tolist()
    x_sequences = []
    for sequence in tqdm(sequences, desc='Tokenizing sequences for ProSST modeling', disable=not verbose):
        x_sequence = []
        for aa in sequence:
            x_sequence.append(vocab[aa])
        x_sequences.append(x_sequence)
    return torch.Tensor(x_sequences).to(torch.int)


def get_logits_from_full_seqs(
        xs, 
        model, 
        input_ids, 
        attention_mask, 
        structure_input_ids,
        train: bool = False,
        verbose: bool = True,
        device: str | None = None
):
    if device is None:
        device = get_device()
    model = model.to(device)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    structure_input_ids = structure_input_ids.to(device)
    if train:
        outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                ss_input_ids=structure_input_ids
        )
    else:
        with torch.no_grad():
            outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    ss_input_ids=structure_input_ids
            )
    logits = torch.log_softmax(outputs.logits[:, 1:-1], dim=-1).squeeze()
    for i_s, sequence in enumerate(
        tqdm(
            xs, disable=not verbose, 
            desc='ProSST inference: getting sequence logits'
        )
    ):
        for i_aa, x_aa in enumerate(sequence):
            if i_aa == 0:
                seq_log_probs = logits[i_aa, x_aa].reshape(1)
            else:
                seq_log_probs = torch.cat(
                    (seq_log_probs, logits[i_aa, x_aa].reshape(1)), 0)
        if i_s == 0:
            log_probs = torch.sum(torch.Tensor(seq_log_probs)).reshape(1)
        else:
            log_probs = torch.cat((
                log_probs, 
                torch.sum(torch.Tensor(seq_log_probs)).reshape(1)
                ), 0
            )
    return log_probs


def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)


def load_model(model, filename):
    logger.info(f'Loading best model: {os.path.abspath(filename)}...')
    model.load_state_dict(torch.load(filename, weights_only=True))


def prosst_train(
        x_sequence_batches, score_batches, loss_fn, model, optimizer,  
        input_ids, attention_mask, structure_input_ids,
        n_epochs=3, device: str | None = None, seed: int | None = None, early_stop: int = 50):
    if seed is not None:
        torch.manual_seed(seed)
    if device is None:
        device = get_device()
    logger.info(f"ProSST training using {device.upper()} device "
          f"(N_Train={len(torch.flatten(score_batches))})...")
    x_sequence_batches = x_sequence_batches.to(device)
    score_batches = score_batches.to(device)
    pbar_epochs = tqdm(range(1, n_epochs + 1))
    epoch_spearman_1 = 0.0
    did_not_improve_counter = 0
    best_model = None
    best_model_epoch = np.nan
    best_model_perf = np.nan
    os.makedirs('model_saves', exist_ok=True)
    for epoch in pbar_epochs:
        if epoch == 0:
            pbar_epochs.set_description(f'Epoch {epoch}/{n_epochs}')
        model.train()
        y_preds_detached = []
        pbar_batches = tqdm(zip(x_sequence_batches, score_batches), 
                            total=len(x_sequence_batches), leave=False)
        for batch, (seqs_b, scores_b) in enumerate(pbar_batches):
            y_preds_b = get_logits_from_full_seqs(
                seqs_b, model, input_ids, attention_mask, structure_input_ids, 
                train=True, verbose=False
            )
            y_preds_detached.append(y_preds_b.detach().cpu().numpy().flatten())
            loss = loss_fn(scores_b, y_preds_b)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            pbar_batches.set_description(
                f"Epoch: {epoch}. Loss: {loss.detach():>1f}  "
                f"[batch: {batch+1}/{len(x_sequence_batches)} | "
                f"sequence: {(batch + 1) * len(seqs_b):>5d}/{len(x_sequence_batches) * len(seqs_b)}]  "
            )
        epoch_spearman_2 = spearmanr(score_batches.cpu().numpy().flatten(), 
                                     np.array(y_preds_detached).flatten())[0]
        if epoch_spearman_2 == np.nan:
            raise SystemError(
                f"No correlation between Y_true and Y_pred could be computed...\n"
                f"Y_true: {score_batches.cpu().numpy().flatten()}, "
                f"Y_pred: {np.array(y_preds_detached)}"
            )
        if epoch_spearman_2 > epoch_spearman_1:
            if best_model is not None:
                if os.path.isfile(best_model):
                    os.remove(best_model)
            did_not_improve_counter = 0
            best_model_epoch = epoch
            best_model_perf = epoch_spearman_2
            best_model = (
                f"model_saves/Epoch{epoch}-Ntrain{len(score_batches.cpu().numpy().flatten())}"
                f"-SpearCorr{epoch_spearman_2:.3f}.pt"
            )
            checkpoint(model, best_model)
            epoch_spearman_1 = epoch_spearman_2
            #logger.info(f"Saved current best model as {best_model}")
        else:
            did_not_improve_counter += 1
            if did_not_improve_counter >= early_stop:
                logger.info(f'\nEarly stop at epoch {epoch}...')
                break
        loss_total = loss_fn(
            torch.flatten(score_batches).to('cpu'), 
            torch.flatten(torch.Tensor(np.array(y_preds_detached).flatten()))
        )
        pbar_epochs.set_description(
            f'Epoch {epoch}/{n_epochs} [SpearCorr: {epoch_spearman_2:.3f}, Loss: {loss_total:.3f}] '
            f'(Best epoch: {best_model_epoch}: {best_model_perf:.3f})')
    try:
        logger.info(f"Loading best model as {best_model}...")
    except UnboundLocalError:
        raise RuntimeError
    load_model(model, best_model)
    y_preds_train = get_logits_from_full_seqs(
        x_sequence_batches.flatten(start_dim=0, end_dim=1), 
        model, input_ids, attention_mask, structure_input_ids, train=False, verbose=False)
    #logger.info(f"Train-->Train Performance (N={len(score_batches.cpu().flatten())}): "
    #            f"{spearmanr(score_batches.cpu().flatten(), y_preds_train.cpu())[0]:.3f}")
    return y_preds_train.cpu()


def get_prosst_models():
    prosst_base_model = AutoModelForMaskedLM.from_pretrained(
        "AI4Protein/ProSST-2048", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        "AI4Protein/ProSST-2048", trust_remote_code=True)
    peft_config = LoraConfig(r=8, target_modules=["query", "value"])
    prosst_lora_model = get_peft_model(prosst_base_model, peft_config)
    optimizer = torch.optim.Adam(prosst_lora_model.parameters(), lr=0.01)  
    return prosst_base_model, prosst_lora_model, tokenizer, optimizer


def get_structure_quantizied(pdb_file, tokenizer, wt_seq):
    structure_sequence = PdbQuantizer()(pdb_file=pdb_file)
    structure_sequence_offset = [i + 3 for i in structure_sequence]
    tokenized_res = tokenizer([wt_seq], return_tensors='pt')
    input_ids = tokenized_res['input_ids']
    attention_mask = tokenized_res['attention_mask']
    structure_input_ids = torch.tensor([1, *structure_sequence_offset, 2], 
                                       dtype=torch.long).unsqueeze(0)
    return input_ids, attention_mask, structure_input_ids


def prosst_setup(wt_seq, pdb_file, sequences, device: str | None = None):
    if wt_seq is None:
        raise SystemError(
            "Running ProSST requires a wild-type sequence "
            "FASTA file input for embedding sequences! "
            "Specify a FASTA file with the --wt flag."
        )
    if pdb_file is None:
        raise SystemError(
            "Running ProSST requires a PDB file input "
            "for embedding sequences! Specify a PDB file "
            "with the --pdb flag."
        )

    pdb_seq = str(list(SeqIO.parse(pdb_file, "pdb-atom"))[0].seq)
    assert wt_seq == pdb_seq, (
        f"Wild-type sequence is not matching PDB-extracted sequence:"
        f"\nWT sequence:\n{wt_seq}\nPDB sequence:\n{pdb_seq}"
    )
    prosst_base_model, prosst_lora_model, prosst_tokenizer, prosst_optimizer = get_prosst_models()
    prosst_vocab = prosst_tokenizer.get_vocab()
    prosst_base_model = prosst_base_model.to(device)
    prosst_optimizer = torch.optim.Adam(prosst_lora_model.parameters(), lr=0.0001)
    input_ids, prosst_attention_mask, structure_input_ids = get_structure_quantizied(
        pdb_file, prosst_tokenizer, wt_seq)
    x_llm_train_prosst = prosst_tokenize_sequences(
        sequences=sequences, vocab=prosst_vocab)
    llm_dict_prosst = {
        'prosst': {
            'llm_base_model': prosst_base_model,
            'llm_model': prosst_lora_model,
            'llm_optimizer': prosst_optimizer,
            'llm_train_function': prosst_train,
            'llm_inference_function': get_logits_from_full_seqs,
            'llm_loss_function': corr_loss,
            'x_llm' : x_llm_train_prosst,
            'llm_attention_mask': prosst_attention_mask,
            'llm_vocab': prosst_vocab,
            'input_ids': input_ids,
            'structure_input_ids': structure_input_ids,
            'llm_tokenizer': prosst_tokenizer
        }
    }
    return llm_dict_prosst
