#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on 05 October 2020
# @author: Niklas Siedhoff, Alexander-Maurice Illig
# <n.siedhoff@biotec.rwth-aachen.de>, <a.illig@biotec.rwth-aachen.de>
# PyPEF - Pythonic Protein Engineering Framework
# Released under Creative Commons Attribution-NonCommercial 4.0 International Public License (CC BY-NC 4.0)
# For more information about the license see https://creativecommons.org/licenses/by-nc/4.0/legalcode

# PyPEF – An Integrated Framework for Data-Driven Protein Engineering
# Journal of Chemical Information and Modeling, 2021
# https://doi.org/10.1021/acs.jcim.1c00099
# Niklas E. Siedhoff1,§, Alexander-Maurice Illig1,§, Ulrich Schwaneberg1,2, Mehdi D. Davari1,*
# 1Institute of Biotechnology, RWTH Aachen University, Worringer Weg 3, 52074 Aachen, Germany
# 2DWI-Leibniz Institute for Interactive Materials, Forckenbeckstraße 50, 52074 Aachen, Germany
# *Corresponding author
# §Equal contribution

"""
Modules for creating learning and validation sets
from input CSV file (with value separators sep=',' or sep=';')
having the CSV-format

HEADER_VARIANTS;HEADER_VARIANTS_FITNESS
VARIANT_1;FITNESS_VALUE_1
VARIANT_2;FITNESS_VALUE_2
...

according to the self devised learning and validation set convention
> VARIANT_NAME_1
; FITNESS_1
VARIANT_SEQUENCE_1
> VARIANT_NAME_2
; FITNESS_2
VARIANT_SEQUENCE_2
...
"""

import numpy as np
import random
import pandas as pd
import re
import glob


def get_wt_sequence(sequence_file):
    """
    Gets wild-type sequence from defined input file (can be pure sequence or fasta style)
    """
    # In wild-type sequence .fa file one has to give the sequence of the studied peptide/protein (wild-type)
    # If no .csv is defined is input this tries to find and select any .fa file
    # (risk: could lead to errors if several .fa files exist in folder)
    if sequence_file is None or sequence_file not in glob.glob(sequence_file):
        try:
            types = ('wt_sequence.*', 'wild_type_sequence.*', '*.fa')  # the tuple of file types
            files_grabbed = []
            for files in types:
                files_grabbed.extend(glob.glob(files))
            for i, file in enumerate(files_grabbed):
                if i == 0:
                    sequence_file = file
            if len(files_grabbed) == 0:
                raise FileNotFoundError("Found no input wild-type fasta sequence file (.fa) in current directory!")
            print('Did not find (specified) WT-sequence file! Used wild-type sequence file instead:'
                  ' {}.'.format(str(sequence_file)))
        except NameError:
            raise NameError("Found no input wild-type fasta sequence file (.fa) in current directory!")
    wild_type_sequence = ""
    with open(sequence_file, 'r') as sf:
        for lines in sf.readlines():
            if lines.startswith(">"):
                continue
            lines = ''.join(lines.split())
            wild_type_sequence += lines
    return wild_type_sequence


def csv_input(csv_file):
    """
    Gets input data from defined .csv file (that contains variant names and fitness labels)
    """
    if csv_file is None or csv_file not in glob.glob(csv_file):
        for i, file in enumerate(glob.glob('*.csv')):
            if file.endswith('.csv'):
                if i == 0:
                    csv_file = file
                    print('Did not find (specified) csv file! Used csv input file instead: {}.'.format(str(csv_file)))
        if len(glob.glob('*.csv')) == 0:
            raise FileNotFoundError('Found no input .csv file in current directory.')
    return csv_file


def drop_rows(csv_file, amino_acids, threshold_drop):
    """
    Drops rows from .csv data if below defined fitness threshold or if
    amino acid/variant name is unknown or if fitness label is not a digit.
    """
    separator = ';'
    try:
        df_raw = pd.read_csv(csv_file, sep=separator, usecols=[0, 1])
    except ValueError:
        separator = ','
        df_raw = pd.read_csv(csv_file, sep=separator, usecols=[0, 1])

    label = df_raw.iloc[:, 1]
    sequence = df_raw.iloc[:, 0]

    dropping_rows = []

    for i, row in enumerate(label):
        try:
            row = float(row)
            if row < threshold_drop:
                dropping_rows.append(i)
        except ValueError:
            dropping_rows.append(i)

    for i, variant in enumerate(sequence):
        try:
            if '/' in variant:
                m = re.split(r'/', variant)
                for a, splits in enumerate(m):
                    if splits[0].isdigit() and variant[-1] in amino_acids:
                        continue
                    elif splits[0] not in amino_acids or splits[-1] not in amino_acids:
                        if i not in dropping_rows:
                            dropping_rows.append(i)
                            # print('Does not know this definition of amino acid substitution: Variant:', variant)
            else:
                if variant[0].isdigit() and variant[-1] in amino_acids:
                    continue
                elif variant[0] not in amino_acids or variant[-1] not in amino_acids:
                    dropping_rows.append(i)
                    # print('Does not know this definition of amino acid substitution: Variant:', variant)
        except TypeError:
            raise TypeError('You might consider checking the input .csv for empty first two columns,'
                            ' e.g. in the last row.')

    print('No. of dropped rows: {}.'.format(len(dropping_rows)), 'Total given variants '
                                                                 '(if provided plus WT): {}'.format(len(df_raw)))

    df = df_raw.drop(dropping_rows)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def get_variants(df, amino_acids, wild_type_sequence):
    """
    Gets variants and divides and counts the variant data for single substituted and higher substituted variants.
    Raises NameError if variant naming is not matching the given wild-type sequence, e.g. if variant A17C would define
    a substitution at residue Ala-17 to Cys but the wild-type sequence has no Ala at position 17.
    """
    x = df.iloc[:, 0]
    y = df.iloc[:, 1]
    wt_position = False
    single_variants, higher_variants, index_higher, index_lower, higher_values, single_values = [], [], [], [], [], []
    single, double, triple, quadruple, quintuple, sextuple, septuple,\
    octuple, nonuple, decuple, higher = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    for i, variant in enumerate(x):
        if '/' in variant:
            count = variant.count('/')
            if count == 1:
                double += 1
            elif count == 2:
                triple += 1
            elif count == 3:
                quadruple += 1
            elif count == 4:
                quintuple += 1
            elif count == 5:
                sextuple += 1
            elif count == 6:
                septuple += 1
            elif count == 7:
                octuple += 1
            elif count == 8:
                nonuple += 1
            elif count == 9:
                decuple += 1
            else:
                higher += 1
            m = re.split(r'/', variant)
            for a, splits in enumerate(m):
                if splits[0].isdigit() or splits[0] in amino_acids and splits[-1] in amino_acids:
                    new = int(re.findall(r'\d+', splits)[0])
                    if splits[0] in amino_acids:
                        if splits[0] != wild_type_sequence[new - 1]:
                            raise NameError('Position of amino acids in given sequence does not match the given '
                                            'positions in the input data! E.g. see position {} and position {} being {}'
                                            ' in the given sequence'.format(variant, new, wild_type_sequence[new - 1]))
                    higher_var = wild_type_sequence[new - 1] + str(new) + str(splits[-1])
                    m[a] = higher_var
                    if a == len(m) - 1:
                        higher_variants.append(m)
                        if i not in index_higher:
                            index_higher.append(i)
        else:
            single += 1
            if variant.upper() == 'WT' or variant.upper() == 'WILD_TYPE':
                wt_position = i

                continue

            elif variant[0].isdigit() or variant[0] in amino_acids and variant[-1] in amino_acids:
                try:
                    num = int(re.findall(r'\d+', variant)[0])
                except IndexError:
                    raise IndexError('Wrong input format. Please check if the input CSV corresponds to the '
                                     'required input style (while the wild-type protein must be designated as \'WT\').')
                if variant[0] in amino_acids:
                    try:
                        if variant[0] != wild_type_sequence[num - 1]:
                            raise NameError('Position of amino acids in given sequence does not match the given '
                                            'positions in the input data! E.g. see position {} and position {} being {}'
                                            ' in the given sequence.'.format(variant, num, wild_type_sequence[num - 1]))
                    except IndexError:
                        raise IndexError("Found variant sequence position {} in data which "
                                         "is out of range of wild-type sequence length.".format(str(num)))
                try:
                    full_variant = wild_type_sequence[num - 1] + str(num) + str(variant[-1])
                except IndexError:
                    raise IndexError("Found variant sequence position {} in data which "
                                     "is out of range of wild-type sequence length.".format(str(num)))
                single_variants.append([full_variant])
                if i not in index_lower:
                    index_lower.append(i)
    print('Single (for mklsvs if provided plus WT): {}.'.format(single), 'Double: {}.'.format(double), 'Triple: {}.'.format(triple),
          'Quadruple: {}.'.format(quadruple), 'Quintuple: {}.'.format(quintuple), 'Sextuple: {}.'.format(sextuple),
          'Septuple: {}.'.format(septuple), 'Octuple: {}.'.format(octuple), 'Nonuple: {}.'.format(nonuple),
          'Decuple: {}.'.format(decuple), 'Higher: {}.'.format(higher))
    for vals in y[index_higher]:
        higher_values.append(vals)
    for vals in y[index_lower]:
        single_values.append(vals)
    if wt_position:
        single_variants.append(['WT'])
        single_values.append(y[wt_position])

    single_variants, single_values = tuple(single_variants), tuple(single_values)
    higher_variants, higher_values = tuple(higher_variants), tuple(higher_values)

    return single_variants, single_values, higher_variants, higher_values


def make_sub_ls_vs(single_variants, single_values, higher_variants, higher_values, directed_evolution=False):
    """
    Creates learning and validation sets, fills learning set with single substituted variants and splits
    rest (higher substituted) for learning and validation sets: 3/4 to LS and 1/4 to VS
    """
    print('No. of single substituted variants (if provided plus WT): {}.'.format(len(single_variants)),
          'No. of values: {}'.format(len(single_values)))
    print('No. of higher substituted variants: {}.'.format(len(higher_variants)),
          'No. of values: {}'.format(len(higher_values)))

    if len(single_values) != len(single_variants):
        print('Error due to different lengths for given variants and label!'
              ' No. of single substituted variants: {}.'.format(len(single_variants)),
              ' Number of given values: {}.'.format(len(single_values)))

    if len(higher_values) != len(higher_variants):
        print('Error due to different lengths for given variants and label! No. of higher subst. variants: {}.'
              .format(len(higher_variants)), ' Number of given values: {}.'.format(len(higher_values)))

    # 1. CREATION OF LS AND VS SPLIT FOR SINGLE FOR LS AND HIGHER VARIANTS FOR VS
    sub_ls = list(single_variants)  # Substitutions of LS
    val_ls = list(single_values)    # Values of LS

    sub_vs = []                     # Substitutions of VS
    val_vs = []                     # Values of VS

    if directed_evolution is False:
        for i in range(len(higher_variants)):
            if len(higher_variants) < 6:  # if less than 6 higher variants all higher variants are appended to VS
                sub_vs.append(higher_variants[i])
                val_vs.append(higher_values[i])
            elif (i % 3) == 0 and i != 0:  # 1/4 of higher variants to VS, 3/4 to LS: change here for LS/VS ratio change
                sub_vs.append(higher_variants[i])
                val_vs.append(higher_values[i])
            else:                       # 3/4 to LS
                sub_ls.append(higher_variants[i])
                val_ls.append(higher_values[i])

    return sub_ls, val_ls, sub_vs, val_vs


def make_sub_ls_vs_randomly(single_variants, single_values, higher_variants, higher_values):
    """
    Creation of learning set and validation set by randomly splitting sets
    """
    length = len(single_variants) + len(higher_variants)
    range_list = np.arange(0, length)

    vs = []
    ls = []
    while len(ls) < length * 4 // 5:
        random_num = random.choice(range_list)
        if random_num not in ls:
            ls.append(random_num)

    for j in range_list:
        if j not in ls:
            vs.append(j)

    combined = single_variants + higher_variants  # substitutions
    combined2 = single_values + higher_values  # values

    sub_ls = []
    val_ls = []
    tot_sub_ls, tot_val_ls = [], []
    tot_sub_vs, tot_val_vs = [], []

    for i in ls:
        sub_ls.append(combined[i])
        val_ls.append(combined2[i])

    sub_vs = []
    val_vs = []
    for j in vs:
        sub_vs.append(combined[j])
        val_vs.append(combined2[j])

    for subs in sub_ls:
        for subs2 in sub_vs:
            if subs == subs2:
                print('\n<Warning> LS and VS overlap for: {} - You might want to consider checking the provided '
                      'datasets for multiple entries'.format(subs), end=' ')

    tot_sub_ls.append(sub_ls)
    tot_val_ls.append(val_ls)
    tot_sub_vs.append(sub_vs)
    tot_val_vs.append(val_vs)

    return tot_sub_ls[0], tot_val_ls[0], tot_sub_vs[0], tot_val_vs[0]


def make_fasta_ls_vs(filename, wt, sub, val):  # sub = substitution, val = value
    """
    Creates learning and validation sets (.fasta style files)
    """
    myfile = open(filename, 'w')
    for i, var in enumerate(sub):  # var are lists of (single or multiple) substitutions
        temp = list(wt)
        name = ''
        separation = 0
        if var == ['WT']:
            name = 'WT'
        else:
            for single_var in var:  # single entries of substitution list
                position_index = int(str(single_var)[1:-1]) - 1
                new_amino_acid = str(single_var)[-1]
                temp[position_index] = new_amino_acid
                # checking if multiple entries are inside list
                if separation == 0:
                    name += single_var
                else:
                    name += '/' + single_var
                separation += 1
        print('>', name, file=myfile)
        print(';', val[i], file=myfile)
        print(''.join(temp), file=myfile)
    myfile.close()
