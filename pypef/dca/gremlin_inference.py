#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on 05 October 2020
# @authors: Niklas Siedhoff, Alexander-Maurice Illig
# @contact: <niklas.siedhoff@rwth-aachen.de>
# PyPEF - Pythonic Protein Engineering Framework
# https://github.com/niklases/PyPEF
# Licensed under Creative Commons Attribution-ShareAlike 4.0 International Public License (CC BY-SA 4.0)
# For more information about the license see https://creativecommons.org/licenses/by-nc/4.0/legalcode

# PyPEF – An Integrated Framework for Data-Driven Protein Engineering
# Journal of Chemical Information and Modeling, 2021, 61, 3463-3476
# https://doi.org/10.1021/acs.jcim.1c00099

"""
Code taken from GREMLIN repository available at https://github.com/sokrypton/GREMLIN_CPP/
and adapted (put functions into a class termed GREMLIN) and used under the
"THE BEER-WARE LICENSE" (Revision 42):
 ----------------------------------------------------------------------------------------
 "THE BEER-WARE LICENSE" (Revision 42):
 <so@fas.harvard.edu> wrote this file.  As long as you retain this notice you
 can do whatever you want with this stuff. If we meet some day, and you think
 this stuff is worth it, you can buy me a beer in return.  Sergey Ovchinnikov
 ----------------------------------------------------------------------------------------
--> Thanks for sharing the great code, I will gladly provide you a beer or two. (Niklas)
Code mainly taken from
https://github.com/sokrypton/GREMLIN_CPP/blob/master/GREMLIN_TF.ipynb

References:
[1] Kamisetty, H., Ovchinnikov, S., & Baker, D.
    Assessing the utility of coevolution-based residue-residue contact predictions in a
    sequence- and structure-rich era.
    Proceedings of the National Academy of Sciences, 2013, 110, 15674-15679
    https://www.pnas.org/doi/10.1073/pnas.1314045110
[2] Balakrishnan, S., Kamisetty, H., Carbonell, J. G., Lee, S.-I., & Langmead, C. J.
    Learning generative models for protein fold families.
    Proteins, 79(4), 2011, 1061-78.
    https://doi.org/10.1002/prot.22934
[3] Ekeberg, M., Lövkvist, C., Lan, Y., Weigt, M., & Aurell, E.
    Improved contact prediction in proteins: Using pseudolikelihoods to infer Potts models.
    Physical Review E, 87(1), 2013, 012707. doi:10.1103/PhysRevE.87.012707
    https://doi.org/10.1103/PhysRevE.87.012707
"""

from __future__ import annotations

import logging
logger = logging.getLogger('pypef.dca.gremlin_inference')

import os
from os import mkdir, PathLike
import pickle
import numpy as np
import matplotlib.pyplot as plt
from Bio import AlignIO
from scipy.spatial.distance import pdist, squareform
from scipy.special import logsumexp
from scipy.stats import boxcox
import pandas as pd
from tqdm import tqdm
import torch


class GREMLIN:
    """
    GREMLIN model in Torch.
    """
    def __init__(
            self,
            alignment: str | PathLike,
            char_alphabet: str = "ARNDCQEGHILKMFPSTWYV-",
            wt_seq=None,
            offset=0,
            optimize=True,
            gap_cutoff=0.5,
            eff_cutoff=0.8,
            opt_iter=100,
            max_msa_seqs: int | None = 10000,
            seqs: list[str] | np.ndarray[str] | None =None,
            device: str | None = None
    ):
        """
        Alphabet char order in GREMLIN: "ARNDCQEGHILKMFPSTWYV-".
        gap_cutoff = 0.5 and eff_cutoff = 0.8 is proposed by Hopf et al.;
        here, by default all columns are used (gap_cutoff >= 1.0 and eff_cutoff > 1.0).
        v_ini represent the weighted frequency of each amino acid at each position, i.e.,
        np.log(np.sum(onehot_cat_msa.T * self.msa_weights, -1).T + pseudo_count).
        """
        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        self.device = device    
        logger.info(f'Using {self.device.upper()} for GREMLIN computations...')   
        self.char_alphabet = char_alphabet
        self.allowed_chars = "ARNDCQEGHILKMFPSTWYV-"
        self.allowed_chars += self.allowed_chars.lower()
        self.offset = offset
        self.gap_cutoff = gap_cutoff
        self.eff_cutoff = eff_cutoff
        self.opt_iter = opt_iter
        self.gaps_1_indexed = None
        if max_msa_seqs == None:
            self.max_msa_seqs = 1E9
        else:
            self.max_msa_seqs = max_msa_seqs
        self.states = len(self.char_alphabet)
        logger.info('Loading MSA...')
        if seqs is None:
            self.seqs, self.seq_ids = self.get_sequences_from_msa(alignment)
        else:
            self.seqs = seqs
            self.seq_ids = np.array([n for n in range(len(self.seqs))])
        logger.info(f'Found {len(self.seqs)} sequences in the MSA...')
        self.msa_ori = self.get_msa_ori()
        logger.info(f'MSA shape: {np.shape(self.msa_ori)}')
        self.n_col_ori = self.msa_ori.shape[1]
        if wt_seq is not None:
            self.wt_seq = wt_seq
        else:  # Taking the first sequence in the MSA as wild type sequence
            self.wt_seq = "".join([self.char_alphabet[i] for i in self.msa_ori[0]])
            logger.info(f"No wild-type sequence provided: The first sequence "
                        f"in the MSA is considered the wild-type sequence "
                        f"(Length: {len(self.wt_seq)}):\n{self.wt_seq}\n")
        if len(self.wt_seq) != self.n_col_ori:
            raise SystemError(f"Length of (provided) wild-type sequence ({len(self.wt_seq)}) "
                              f"does not match number of MSA columns ({self.n_col_ori}), "
                              f"i.e., common MSA sequence length.")
        logger.info('Filtering gaps...')
        self.msa_trimmed, self.v_idx, self.w_idx, self.w_rel_idx, self.gaps = self.filt_gaps(self.msa_ori)
        logger.info('Getting effective sequence weights...')
        self.msa_weights = self.get_eff_msa_weights(self.msa_trimmed)
        self.n_eff = np.sum(self.msa_weights)
        self.n_row = self.msa_trimmed.shape[0]
        self.n_col = self.msa_trimmed.shape[1]
        logger.info('Initializing v and W terms based on MSA frequencies...')
        self.v_ini, self.w_ini, self.aa_counts = self.initialize_v_w(remove_gap_entries=False)
        self.aa_freqs = self.aa_counts / self.n_row
        self.optimize = optimize
        if self.optimize:
            self.run_optimization()
        self.x_wt = self.collect_encoded_sequences(np.atleast_1d(self.wt_seq))

    def get_sequences_from_msa(self, msa_file: str):
        """
        "Get_Sequences" reads (learning and test) .fasta and
        .fasta-like ".fasl" format files and extracts the name,
        the target value and the sequence of the protein.
        Only takes one-liner sequences for correct input.
        See example directory for required fasta file format.
        Make sure every marker (> and ;) is seperated by a
        space ' ' from the value respectively name.
        Trimming MSA sequences starting at offset.

        msa_file: str
            Path to MSA in FASTA or A2M format.
        """
        sequences = []
        seq_ids = []
        with open(msa_file, 'r') as fh:
            alignment = AlignIO.read(fh, "fasta")
        for record in alignment:
            sequences.append(str(record.seq))
            seq_ids.append(str(record.id))
        assert len(sequences) == len(seq_ids), f"{len(sequences)}, {len(seq_ids)}"
        return np.array(sequences), np.array(seq_ids)

    def a2n_dict(self):
        """
        Convert alphabet to numerical integer values, e.g.:
        {"A": 0, "C": 1, "D": 2, ...}
        """
        a2n = {}
        for a, n in zip(self.char_alphabet, range(self.states)):
            a2n[a] = n
        return a2n

    def aa2int(self, aa):
        """
        convert single aa into numerical integer value, e.g.:
        "A" -> 0 or "-" to 21 dependent on char_alphabet
        """
        a2n = self.a2n_dict()
        if aa in a2n:
            return a2n[aa]
        else:  # for unknown characters insert Gap character
            return a2n['-']

    def seq2int(self, aa_seqs):
        """
        Convert a single sequence or a list of sequences 
        into a list of integer sequences, e.g.:
        ["ACD","EFG"] -> [[0,4,3], [6,13,7]]
        """
        aa_seqs = np.atleast_1d(aa_seqs)
        return np.array([[self.aa2int(aa) for aa in seq] for seq in aa_seqs])

    @property
    def get_v_idx_w_idx(self):
        return self.v_idx, self.w_idx

    def get_msa_ori(self):
        """
        Converts list of sequences to MSA.
        Also checks for unknown amino acid characters 
        and removes those sequences from the MSA.
        """
        msa_ori = []
        for i, (seq, _seq_id) in enumerate(zip(self.seqs, self.seq_ids)):
            if i < self.max_msa_seqs:
                msa_ori.append([self.aa2int(aa.upper()) for aa in seq])
            else:
                logger.info(f'Reached max. number of MSA sequences ({self.max_msa_seqs})...')
                break
        msa_ori = np.array(msa_ori)
        return msa_ori

    def filt_gaps(self, msa_ori):
        """Filters alignment to remove gappy positions"""
        tmp = (msa_ori == self.states - 1).astype(float)
        non_gaps = np.where(np.sum(tmp.T, -1).T / msa_ori.shape[0] < self.gap_cutoff)[0]
        gaps = np.where(np.sum(tmp.T, -1).T / msa_ori.shape[0] >= self.gap_cutoff)[0]
        self.gaps_1_indexed = [int(g + 1) for g in gaps]
        logger.info(f'Gap positions (removed from MSA; 1-indexed):\n{self.gaps_1_indexed}')
        ncol_trimmed = len(non_gaps)
        logger.info(f'Positions remaining: {ncol_trimmed} of {np.shape(msa_ori)[1]} '
                    f'({(ncol_trimmed / np.shape(msa_ori)[1]) * 100 :.2f}%)')
        v_idx = non_gaps
        w_idx = v_idx[np.stack(np.triu_indices(ncol_trimmed, 1), -1)]
        w_rel_idx = np.stack(np.triu_indices(ncol_trimmed, 1), -1)
        return msa_ori[:, non_gaps], v_idx, w_idx, w_rel_idx, gaps

    def get_eff_msa_weights(self, msa):
        """Compute effective weight for each sequence"""
        # pairwise identity
        pdistance_msa = pdist(msa, "hamming")  # TODO: to PyTorch?
        msa_sm = 1.0 - squareform(pdistance_msa)
        # weight for each sequence
        msa_w = (msa_sm >= self.eff_cutoff).astype(float)
        msa_w = 1 / np.sum(msa_w, -1)
        return msa_w

    @staticmethod
    def flatten_v_w(v, w):
        return torch.cat((v.flatten(), w.flatten()), 0)


    def opt_adam_step(self, lr=1.0, b1=0.9, b2=0.999):
        """
        Adam optimizer [https://arxiv.org/abs/1412.6980] with first and second moments
            mt          and
            vt          (greek letter nu) at time steps t, respectively.
        Note by GREMLIN authors: this is a modified version of adam optimizer.
        More specifically, we replace "vt" with sum(g*g) instead of (g*g).
        Furthermore, we find that disabling the bias correction
        (b_fix=False) speeds up convergence for our case.
        """
        self.v.retain_grad()
        self.w.retain_grad()
        loss = self.loss(self.v, self.w)
        loss.backward()

        mt_tmp_v = b1 * self.mt_v + (1 - b1) * self.v.grad
        vt_tmp_v = b2 * self.vt_v + (1 - b2) * torch.sum(torch.square(self.v.grad))
        lr_tmp_v = lr / (torch.sqrt(vt_tmp_v) + 1e-8)
        self.v = self.v.add(-lr_tmp_v * mt_tmp_v)

        mt_tmp_w = b1 * self.mt_w + (1 - b1) * self.w.grad
        vt_tmp_w = b2 * self.vt_w + (1 - b2) * torch.sum(torch.square(self.w.grad))
        lr_tmp_w = lr / (torch.sqrt(vt_tmp_w) + 1e-8)
        self.w = self.w.add(-lr_tmp_w * mt_tmp_w)

        self.vt_v = vt_tmp_v
        self.mt_v = mt_tmp_v
        self.vt_w = vt_tmp_w
        self.mt_w = mt_tmp_w

    def sym_w(self, w, device: str | None = None):
        """
        Symmetrize input matrix of shape (x,y,x,y)
        As the full couplings matrix W might/will be slightly "unsymmetrical"
        it will be symmetrized according to one half being "mirrored".
        """
        if device is None:
            device = self.device
        x = w.shape[0]
        w = w * torch.reshape(1 - torch.eye(x), (x, 1, x, 1)).to(device)
        w = w + torch.permute(w, (2, 3, 0, 1))
        return w

    @staticmethod
    def l2_reg(x):
        return torch.sum(torch.square(x))
    
    def loss(self, v, w, device: str | None = None):
        ##############################################################
        # SETUP COMPUTE GRAPH
        ##############################################################
        if device is None:
            device = self.device
        v, w = v.to(device), w.to(device)
        # symmetrize w
        w = self.sym_w(w, device).to(torch.float32)

        ########################################
        # Pseudo-Log-Likelihood
        ########################################
        vw = v + torch.tensordot(self.oh_msa.to(device), w, dims=2)

        # Hamiltonian
        h = torch.sum(torch.mul(self.oh_msa.to(device), vw), dim=(1, 2))
        # partition function Z
        z = torch.sum(torch.logsumexp(vw, dim=2), dim=1)

        # Pseudo-Log-Likelihood
        pll = h - z

        ########################################
        # Regularization
        ########################################
        l2_v = 0.01 * self.l2_reg(v)
        lw_w = 0.01 * self.l2_reg(w) * 0.5 * (self.n_col - 1) * (self.states - 1)

        # loss function to minimize
        loss = (
            -torch.sum(pll * self.msa_weights.to(device)) / 
            torch.sum(self.msa_weights.to(device))
        )
        loss = loss + (l2_v + lw_w) / self.n_eff
        return loss
    
    def _loss(self, decimals=2):
        return  torch.round(
            self.loss(self.v.detach(), self.w.detach(), device='cpu') * self.n_eff, 
            decimals=decimals
        )

    def run_optimization(self):
        """
        For optimization of v and w ADAM is used here (L-BFGS-B not (yet) implemented
        for TF 2.x, e.g. using scipy.optimize.minimize).
        Gaps (char '-' respectively '21') included.
        """
        ##############################################################
        # MINIMIZE LOSS FUNCTION
        ##############################################################
        # initialize V (local fields)
        msa_cat = np.eye(self.states)[self.msa_trimmed]
        pseudo_count = 0.01 * np.log(self.n_eff)
        v_ini = np.log(np.sum(msa_cat.T * self.msa_weights, -1).T + pseudo_count)
        v_ini = v_ini - np.mean(v_ini, -1, keepdims=True)
        self.v = torch.from_numpy(v_ini).to(torch.float32).requires_grad_(True).to(self.device)
        self.w = torch.zeros(
            size=(self.n_col, self.states, self.n_col, self.states)
            ).to(torch.float32).requires_grad_(True).to(self.device)

        self.msa = torch.Tensor(self.msa_trimmed).to(torch.int64).to(self.device)
        self.oh_msa = torch.nn.functional.one_hot(self.msa, self.states).to(torch.float32).to(self.device)
        self.msa_weights = torch.from_numpy(self.msa_weights).to(torch.float32).to(self.device)

        self.mt_v, self.vt_v = torch.zeros_like(self.v), torch.zeros_like(self.v)
        self.mt_w, self.vt_w = torch.zeros_like(self.w), torch.zeros_like(self.w)
        logger.info(f'Initial loss: {self._loss()}')
        for i in range(self.opt_iter):
            self.opt_adam_step()
            try:
                if (i + 1) % int(self.opt_iter / 10) == 0:
                    logger.info(f'Loss step {i + 1}: {self._loss()}')
            except ZeroDivisionError:
                logger.info(f'Loss step {i + 1}: {self._loss()}')
        
        self.v = self.v.detach().cpu().numpy()
        self.w = self.w.detach().cpu().numpy()
        self.vt_v = self.vt_v.detach().cpu().numpy()
        self.mt_v = self.mt_v.detach().cpu().numpy()
        self.vt_w = self.vt_w.detach().cpu().numpy()
        self.mt_w = self.mt_w.detach().cpu().numpy()


    def initialize_v_w(self, remove_gap_entries=True):
        """
        For optimization of v and w ADAM is used here (L-BFGS-B not (yet)
        implemented for TF 2.x, e.g. using scipy.optimize.minimize).
        Gaps (char '-' respectively '21') included.
        """
        w_ini = np.zeros((self.n_col, self.states, self.n_col, self.states))
        onehot_cat_msa = np.eye(self.states)[self.msa_trimmed]
        aa_counts = np.sum(onehot_cat_msa, axis=0)
        pseudo_count = 0.01 * np.log(self.n_eff)
        v_ini = np.log(np.sum(onehot_cat_msa.T * self.msa_weights, -1).T + pseudo_count)
        v_ini = v_ini - np.mean(v_ini, -1, keepdims=True)

        if remove_gap_entries:
            no_gap_states = self.states - 1
            v_ini = v_ini[:, :no_gap_states]
            w_ini = w_ini[:, :no_gap_states, :, :no_gap_states]
            aa_counts = aa_counts[:, :no_gap_states]

        return v_ini, w_ini, aa_counts

    @property
    def get_v_w(self):
        try:
            return self.v, self.w
        except AttributeError:
            raise SystemError(
                "No v and w available, this means GREMLIN "
                "has not been initialized setting optimize to True, "
                "e.g., try GREMLIN('Alignment.fasta', optimize=True)."
            )

    def get_scores(self, seqs, v=None, w=None, v_idx=None, encode=False, h_wt_seq=0.0, recompute_z=False):
        """
        Computes the GREMLIN score for a given sequence or list of sequences.
        """
        if v is None and w is None:
            if self.optimize:
                v = self.v[:, :self.states-1], 
                w = self.w[:, :self.states-1, :, :self.states-1]
            else:
                v, w, _ = self.initialize_v_w(remove_gap_entries=True)
        if v_idx is None:
            v_idx = self.v_idx
        seqs_int = self.seq2int(seqs)

        try:
            if seqs_int.shape[-1] != len(v_idx):  # The input sequence length ({seqs_int.shape[-1]}) 
                # does not match the common gap-trimmed MSA sequence length (len(v_idx)
                seqs_int = seqs_int[..., v_idx]  # Shape matches common MSA sequence length (len(v_idx)) now
        except IndexError:
            raise SystemError(
                "The loaded GREMLIN parameter model does not match the input model "
                "in terms of sequence encoding shape or is a gap-substituted sequence. "
                "E.g., when providing two different DCA models/parameters provided by: "
                "\"-m DCA\" and \"--params GREMLIN\", where -m DCA represents a ml input "
                "model potentially generated using plmc parameters and --params GREMLIN "
                "provides differently encoded sequences generated using GREMLIN."
            )

        # one hot encode
        x = np.eye(self.states)[seqs_int]

        # get non-gap positions
        # no_gap = 1.0 - x[..., -1]

        # remove gap from one-hot-encoding
        x = x[..., :-1]

        # compute score
        vw = v + np.tensordot(x, w, 2)

        # ============================================================================================
        # Note, Z (the partition function) is a constant. In GREMLIN, V, W & Z are estimated using all
        # the original weighted input sequence(s). It is NOT recommended to recalculate z with a
        # different set of sequences. Given the common ERROR of recomputing Z, we include the option
        # to do so, for comparison.
        # ============================================================================================
        h = np.sum(np.multiply(x, vw), axis=-1)

        if encode:
            return h

        if recompute_z:
            z = logsumexp(vw, axis=-1)
            return np.sum((h - z), axis=-1) - h_wt_seq
        else:
            return np.sum(h, axis=-1) - h_wt_seq

    def get_wt_score(self, wt_seq=None, encode=False):
        if wt_seq is None:
            wt_seq = self.wt_seq
        wt_seq = np.array(wt_seq, dtype=str)
        return self.get_scores(wt_seq, encode=encode)

    def collect_encoded_sequences(self, seqs, v=None, w=None, v_idx=None):
        """
        Wrapper function for encoding input sequences using the self.get_scores
        function with encode set to True.
        """
        xs = self.get_scores(seqs, v, w, v_idx, encode=True)
        return xs

    @staticmethod
    def normalize(apc_mat):
        """
        Normalization of APC matrix for getting z-Score matrix
        """
        dim = apc_mat.shape[0]
        apc_mat_flat = apc_mat.flatten()
        x, _ = boxcox(apc_mat_flat - np.amin(apc_mat_flat) + 1.0)
        x_mean = np.mean(x)
        x_std = np.std(x)
        x = (x - x_mean) / x_std
        x = x.reshape(dim, dim)
        return x

    def mtx_gaps_as_zeros(self, gap_reduced_mtx, insert_gap_zeros=True):
        """
        Inserts zeros at gap positions of the (L,L) matrices,
        i.e., raw/apc/zscore matrices.
        """
        if insert_gap_zeros:
            gap_reduced_mtx = list(gap_reduced_mtx)
            mtx_zeroed = []
            c = 0
            for i in range(self.n_col_ori):
                mtx_i = []
                c_i = 0
                if i in self.gaps:
                    mtx_zeroed.append(list(np.zeros(self.n_col_ori)))
                else:
                    for j in range(self.n_col_ori):
                        if j in self.gaps:
                            mtx_i.append(0.0)
                        else:
                            mtx_i.append(gap_reduced_mtx[c][c_i])
                            c_i += 1
                    mtx_zeroed.append(mtx_i)
                    c += 1
            return np.array(mtx_zeroed)

        else:
            return gap_reduced_mtx

    def get_correlation_matrix(self, matrix_type: str = 'apc', insert_gap_zeros=False):
        """
        Requires optimized w matrix (of shape (L, 20, L, 20))
        inputs
        ------------------------------------------------------
        w           : coevolution       shape=(L,A,L,A)
        ------------------------------------------------------
        outputs
        ------------------------------------------------------
        raw         : l2norm(w)         shape=(L,L)
        apc         : apc(raw)          shape=(L,L)
        zscore      : normalize(apc)    shape=(L,L)
        """
        # l2norm of 20x20 matrices (note: gaps already excluded)
        raw = np.sqrt(np.sum(np.square(self.w), (1, 3)))

        # apc (average product correction)
        ap = np.sum(raw, 0, keepdims=True) * np.sum(raw, 1, keepdims=True) / np.sum(raw)
        apc = raw - ap

        if matrix_type == 'apc':
            return self.mtx_gaps_as_zeros(apc, insert_gap_zeros=insert_gap_zeros)
        elif matrix_type == 'raw':
            return self.mtx_gaps_as_zeros(raw, insert_gap_zeros=insert_gap_zeros)
        elif matrix_type == 'zscore' or matrix_type == 'z_score':
            return self.mtx_gaps_as_zeros(self.normalize(apc), insert_gap_zeros=insert_gap_zeros)
        else:
            raise SystemError("Unknown matrix type. Choose between 'apc', 'raw', or 'zscore'.")

    def plot_correlation_matrix(self, matrix_type: str = 'apc', set_diag_zero=True):
        matrix = self.get_correlation_matrix(matrix_type, insert_gap_zeros=True)
        if set_diag_zero:
            np.fill_diagonal(matrix, 0.0)

        _fig, ax = plt.subplots(figsize=(10, 10))

        if matrix_type == 'zscore' or matrix_type == 'z_score':
            ax.imshow(matrix, cmap='Blues', interpolation='none', vmin=1, vmax=3)
        else:
            ax.imshow(matrix, cmap='Blues')
        tick_pos = ax.get_xticks()
        tick_pos = np.array([int(t) for t in tick_pos])
        tick_pos[-1] = matrix.shape[0]
        if tick_pos[2] > 1:
            tick_pos[2:] -= 1
        ax.set_xticks(tick_pos)
        ax.set_yticks(tick_pos)
        labels = [item.get_text() for item in ax.get_xticklabels()]
        try:
            labels = [labels[0]] + [str(int(label) + 1) for label in labels[1:]]
        except ValueError:
            pass
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlim(-1, matrix.shape[0])
        ax.set_ylim(-1, matrix.shape[0])
        plt.title(matrix_type.upper())
        plt.savefig(f'{matrix_type}.png', dpi=500)
        plt.close('all')

    def get_top_coevolving_residues(self, wt_seq=None, min_distance=0, sort_by="apc"):
        if wt_seq is None:
            wt_seq = self.wt_seq
        if wt_seq is None:
            raise SystemError("Getting top co-evolving residues requires "
                              "the wild type sequence as input.")
        raw = self.get_correlation_matrix(matrix_type='raw')
        apc = self.get_correlation_matrix(matrix_type='apc')
        zscore = self.get_correlation_matrix(matrix_type='zscore')

        # Explore top co-evolving residue pairs
        i_rel_idx = self.w_rel_idx[:, 0]
        j_rel_idx = self.w_rel_idx[:, 1]

        apc_flat = []
        zscore_flat = []
        raw_flat = []
        for i, _ in enumerate(i_rel_idx):
            raw_flat.append(raw[i_rel_idx[i]][j_rel_idx[i]])
            apc_flat.append(apc[i_rel_idx[i]][j_rel_idx[i]])
            zscore_flat.append(zscore[i_rel_idx[i]][j_rel_idx[i]])

        i_idx = self.w_idx[:, 0]
        j_idx = self.w_idx[:, 1]

        i_aa = [f"{wt_seq[i]}_{i + 1}" for i in i_idx]
        j_aa = [f"{wt_seq[j]}_{j + 1}" for j in j_idx]

        # load mtx into pandas dataframe
        mtx = {
            "i": i_idx, "j": j_idx, "apc": apc_flat, "zscore": zscore_flat,
            "raw": raw_flat, "i_aa": i_aa, "j_aa": j_aa
        }
        df_mtx = pd.DataFrame(mtx, columns=["i", "j", "apc", "zscore", "raw", "i_aa", "j_aa"])
        df_mtx_sorted = df_mtx.sort_values(sort_by, ascending=False)

        # get contacts with sequence separation > min_distance
        df_mtx_sorted_mindist = df_mtx_sorted.loc[df_mtx['j'] - df_mtx['i'] > min_distance]

        return df_mtx_sorted_mindist


"""
GREMLIN class helper functions below.
"""


def save_gremlin_as_pickle(alignment: str, wt_seq: str, opt_iter: int = 100):
    """
    Function for getting and/or saving (optimized or unoptimized) GREMLIN model
    """
    logger.info(f'Inferring GREMLIN DCA parameters based on the provided MSA...')
    gremlin = GREMLIN(alignment, wt_seq=wt_seq, optimize=True, opt_iter=opt_iter)
    try:
        mkdir('Pickles')
    except FileExistsError:
        pass

    pickle.dump(
        {
            'model': gremlin,
            'model_type': 'GREMLINpureDCA',
            'beta_1': None,
            'beta_2': None,
            'spearman_rho': None,
            'regressor': None
        },
        open('Pickles/GREMLIN', 'wb')
    )
    logger.info(f"Saved GREMLIN model as Pickle file as {os.path.abspath('Pickles/GREMLIN')}...")
    return gremlin


def get_delta_e_statistical_model(
        x_test: np.ndarray,
        x_wt: np.ndarray
):
    """
    Description
    -----------
    Delta_E means difference in evolutionary energy in plmc terms.
    In other words, this is the delta of the sum of Hamiltonian-encoded
    sequences of local fields and couplings of encoded sequence and wild-type
    sequence in GREMLIN terms.

    Parameters
    -----------
    x_test: np.ndarray [2-dim]
        Encoded sequences to be subtracted by x_wt to compute delta E.
    x_wt: np.ndarray [1-dim]
        Encoded wild-type sequence.

    Returns
    -----------
    delta_e: np.ndarray [1-dim]
        Summed subtracted encoded sequences.

    """
    delta_x = np.subtract(x_test, x_wt)
    delta_e = np.sum(delta_x, axis=1)
    return delta_e


def plot_all_corr_mtx(gremlin: GREMLIN):
    gremlin.plot_correlation_matrix(matrix_type='raw')
    gremlin.plot_correlation_matrix(matrix_type='apc')
    gremlin.plot_correlation_matrix(matrix_type='zscore')


def save_corr_csv(gremlin: GREMLIN, min_distance: int = 0, sort_by: str = 'apc'):
    df_mtx_sorted_mindist = gremlin.get_top_coevolving_residues(
        min_distance=min_distance, sort_by=sort_by
    )
    df_mtx_sorted_mindist.to_csv(f"coevolution_{sort_by}_sorted.csv", sep=',')
    logger.info(f"Saved coevolution CSV data as "
                f"{os.path.abspath(f'coevolution_{sort_by}_sorted.csv')}")


def plot_predicted_ssm(gremlin: GREMLIN):
    """
    Function to plot all predicted 19 amino acid substitution 
    effects at all predictable WT/input sequence positions; e.g.: 
    M1A, M1C, M1E, ..., D2A, D2C, D2E, ..., ..., T300V, T300W, T300Y
    """
    wt_sequence = gremlin.wt_seq
    wt_score = gremlin.get_wt_score()[0]
    aas = "".join(sorted(gremlin.char_alphabet.replace("-", "")))
    variantss, variant_sequencess, variant_scoress = [], [], []
    logger.info("Predicting all SSM effects using the unsupervised GREMLIN model...")
    for i, aa_wt in enumerate(tqdm(wt_sequence)):
        variants, variant_sequences, variant_scores = [], [], []
        for aa_sub in aas:
            variant = aa_wt + str(i + 1) + aa_sub
            variant_sequence = wt_sequence[:i] + aa_sub + wt_sequence[i + 1:]
            variant_score = gremlin.get_scores(variant_sequence)[0]
            variants.append(variant)
            variant_sequences.append(variant_sequence)
            variant_scores.append(variant_score - wt_score)
        variantss.append(variants)
        variant_sequencess.append(variant_sequences)
        variant_scoress.append(variant_scores)

    fig, ax = plt.subplots(figsize=(2 * len(wt_sequence) / len(aas), 3))
    ax.imshow(np.array(variant_scoress).T)
    for i_vss, vss in enumerate(variant_scoress):
        for i_vs, vs in enumerate(vss):
            ax.text(
                i_vss, i_vs, 
                f'{variantss[i_vss][i_vs]}\n{round(vs, 1)}', 
                size=1.5, va='center', ha='center'
            )
    ax.set_xticks(
        range(len(wt_sequence)), 
        [f'{aa}{i + 1}' for i, aa in enumerate(wt_sequence)], 
        size=6, rotation=90
    )
    ax.set_yticks(range(len(aas)), aas, size=6)
    plt.tight_layout()
    plt.savefig('SSM_landscape.png', dpi=500)
    pd.DataFrame(
        {
            'Variant': np.array(variantss).flatten(),
            'Sequence': np.array(variant_sequencess).flatten(),
            'Variant_Score': np.array(variant_scoress).flatten()
        }
    ).to_csv('SSM_landscape.csv', sep=',')
    logger.info(f"Saved SSM landscape as {os.path.abspath('SSM_landscape.png')} "
                f"and CSV data as {os.path.abspath('SSM_landscape.csv')}...")
