#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on 05 October 2020
# @authors: Niklas Siedhoff, Alexander-Maurice Illig
# @contact: <n.siedhoff@biotec.rwth-aachen.de>
# PyPEF - Pythonic Protein Engineering Framework
# Released under Creative Commons Attribution-NonCommercial 4.0 International Public License (CC BY-NC 4.0)
# For more information about the license see https://creativecommons.org/licenses/by-nc/4.0/legalcode

# PyPEF – An Integrated Framework for Data-Driven Protein Engineering
# Journal of Chemical Information and Modeling, 2021, 61, 3463-3476
# https://doi.org/10.1021/acs.jcim.1c00099
# Niklas E. Siedhoff1,§, Alexander-Maurice Illig1,§, Ulrich Schwaneberg1,2, Mehdi D. Davari1,*
# 1Institute of Biotechnology, RWTH Aachen University, Worringer Weg 3, 52074 Aachen, Germany
# 2DWI-Leibniz Institute for Interactive Materials, Forckenbeckstraße 50, 52074 Aachen, Germany
# *Corresponding author
# §Equal contribution

import logging
logger = logging.getLogger('pypef.utils.sto2a2m')
import numpy as np
from tqdm import tqdm
from Bio import AlignIO


def convert_sto2a2m(
        sto_file: str,
        inter_gap: float,
        intra_gap: float
):
    """
    Converts alignment in format STO to A2M format.
    Removes specific sequences with inter and/or intra gaps
    over specific thresholds.
    """
    # Generate the a2m output filename
    a2m_file = f"{sto_file.split('.sto')[0]}.a2m"

    # Load the stockholm alignment
    logger.info('Loading MSA in stockholm format...')
    sto_alignment = AlignIO.read(sto_file, 'stockholm')
    logger.info('Trimming MSA...')
    # Save this 'raw' multiple sequence alignment as numpy array 
    raw_msa = []
    for record in tqdm(sto_alignment):
        raw_msa.append(np.array(record.seq))
    raw_msa = np.array(raw_msa)

    # 1st processing step
    # Delete all positions, where WT has a gap to obtain the 'trimmed' MSA
    ungap_pos = np.where(raw_msa[0] == "-")
    msa_trimmed = np.array([np.delete(seq, ungap_pos) for seq in raw_msa])

    # 2nd processing step
    # Remove ("lower") all positions with more than 'inter_gap'*100 % gaps (columnar trimming)
    count_gaps = np.count_nonzero(msa_trimmed == '-', axis=0) / msa_trimmed.shape[0]
    lower = [idx for idx, count in enumerate(count_gaps) if count > inter_gap]
    msa_trimmed_T = msa_trimmed.T
    for idx in lower:
        msa_trimmed_T[idx] = np.char.lower(msa_trimmed_T[idx])
        # replace all columns that are "removed" due to high gap content and have an "-" element by "." 
        msa_trimmed_T[idx] = np.where(msa_trimmed_T[idx] == '-', '.', msa_trimmed_T[idx])
    msa_trimmed_inter_gap = msa_trimmed_T.T

    # 3rd processing step
    # Remove all sequences with more than 'intra_gap'*100 % gaps (line trimming)
    target_len = len(msa_trimmed_inter_gap[0])
    gap_content = (np.count_nonzero(msa_trimmed_inter_gap == "-", axis=1) + np.count_nonzero(
        msa_trimmed_inter_gap == ".", axis=1)) / target_len
    delete = np.where(gap_content > intra_gap)[0]
    msa_final = np.delete(msa_trimmed_inter_gap, delete, axis=0)
    seqs_cls = [seq_cls for idx, seq_cls in enumerate(sto_alignment) if not idx in delete]
    chunk_size = 60
    with open(a2m_file, 'w') as f:
        for i, (seq, seq_cls) in enumerate(zip(msa_final, seqs_cls)):
            if i == 0:
                f.write(f'>TARGET_SEQ\n')
            else:
                f.write('>' + seq_cls.id + '\n')
            for chunk in [seq[x:x + chunk_size] for x in range(0, len(seq), chunk_size)]:
                f.write("".join(chunk) + '\n')

    # Get number of sequences and effective sites in the alignment
    n_seqs = msa_final.shape[0]
    n_sites = sum(1 for char in msa_final[0] if char.isupper())
    logger.info(f'Generated trimmed MSA {a2m_file} in A2M format:\n'
                f'No. of sequences: {n_seqs}\n'
                f'No. of effective sites: {n_sites} (out of {target_len} sites)\n'
                f'-le --lambdae: {0.2 * (n_sites - 1):.1f}')

    return n_seqs, n_sites, target_len
