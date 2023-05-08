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

"""
Modules for creating training and test sets
from input CSV file (with value separators sep=',' or sep=';')
having the CSV-format

HEADER_VARIANTS;HEADER_VARIANTS_FITNESS
VARIANT_1;FITNESS_VALUE_1
VARIANT_2;FITNESS_VALUE_2
...

according to the self devised training and test set convention
> VARIANT_NAME_1
; FITNESS_1
VARIANT_SEQUENCE_1
> VARIANT_NAME_2
; FITNESS_2
VARIANT_SEQUENCE_2
...
"""

import logging
logger = logging.getLogger('pypef.utils.learning_test_sets')

import numpy as np
import random
import pandas as pd
import re


def get_wt_sequence(sequence_fasta):
    """
    Gets wild-type sequence from defined input file (can be pure sequence or fasta style)
    """
    wild_type_sequence = ""
    try:
        with open(sequence_fasta, 'r') as sf:
            for lines in sf.readlines():
                if lines.startswith(">"):
                    continue
                lines = ''.join(lines.split())
                wild_type_sequence += lines
    except FileNotFoundError:
        raise FileNotFoundError("Did not find FASTA file. Check/specify input FASTA "
                                "sequence file for getting the wild-type sequence.")
    return wild_type_sequence


def csv_input(csv_file):
    """
    Gets input data from defined .csv file (that contains variant names and fitness labels)
    """
    if csv_file is None:
        raise FileNotFoundError(
            f'Did not find (specified) csv file! '
            f'Used csv input file instead: {str(csv_file)}.'
        )
    return csv_file


def drop_rows(
        csv_file,
        amino_acids,
        threshold_drop
):
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
    except FileNotFoundError:
        raise FileNotFoundError(
            "Specify the input CSV file containing the variant-fitness data. "
            "Required CSV format: variant;fitness.")

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
                elif variant not in ['wt', 'wild_type']:
                    if variant[0] not in amino_acids or variant[-1] not in amino_acids:
                        dropping_rows.append(i)
                        # print('Does not know this definition of amino acid substitution: Variant:', variant)
        except TypeError:
            raise TypeError('You might consider checking the input .csv for empty first two columns,'
                            ' e.g. in the last row.')

    logger.info(f'No. of dropped rows: {len(dropping_rows)}. '
                f'Total given variants (if provided plus WT): {len(df_raw)}')

    df = df_raw.drop(dropping_rows)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def get_variants(
        df,
        amino_acids,
        wild_type_sequence
):
    """
    Gets variants and divides and counts the variant data for single substituted
    and higher substituted variants. Raises NameError if variant naming is not
    matching the given wild-type sequence, e.g. if variant A17C would define
    a substitution at residue Ala-17 to Cys but the wild-type sequence has no Ala
    at position 17.
    """
    x = df.iloc[:, 0]
    y = df.iloc[:, 1]
    wt_position = None
    single_variants, higher_variants, index_higher, index_lower, \
        higher_values, single_values = [], [], [], [], [], []
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
                            raise NameError(
                                'Position of amino acids in given sequence does not match the given '
                                'positions in the input data! E.g. see position {} and position {} being {} '
                                'in the given sequence'.format(variant, new, wild_type_sequence[new - 1])
                            )
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
    logger.info(
        '\nSingle (for mklsts if provided plus WT): {}\nDouble: {}\nTriple: {}\nQuadruple: {}\nQuintuple: {}\n'
        'Sextuple: {}\nSeptuple: {}\nOctuple: {}\nNonuple: {}\nDecuple: {}\nHigher (>Decuple): {}'.format(
            single, double, triple, quadruple, quintuple, sextuple, septuple, octuple, nonuple, decuple, higher
        )
    )
    for vals in y[index_higher]:
        higher_values.append(vals)
    for vals in y[index_lower]:
        single_values.append(vals)
    if wt_position is not None:
        single_variants.append(['WT'])
        single_values.append(y[wt_position])

    single_variants, single_values = tuple(single_variants), tuple(single_values)
    higher_variants, higher_values = tuple(higher_variants), tuple(higher_values)

    return single_variants, single_values, higher_variants, higher_values


def make_sub_ls_ts(
        single_variants,
        single_values,
        higher_variants,
        higher_values,
        directed_evolution=False):
    """
    Creates learning and test sets, fills learning set with single substituted variants and splits
    rest (higher substituted) for learning and test sets: 3/4 to LS and 1/4 to TS
    """
    logger.info(f'No. of single substituted variants (if provided plus WT): {len(single_variants)}.'
                f'No. of values: {len(single_values)}.')
    logger.info(f'No. of higher substituted variants: {len(higher_variants)}. '
                f'No. of values: {len(higher_values)}.')

    if len(single_values) != len(single_variants):
        logger.info(f'Error due to different lengths for given variants and label! '
                    'No. of single substituted variants: {len(single_variants)}. '
                    'Number of given values: {len(single_values)}.')

    if len(higher_values) != len(higher_variants):
        logger.info(f'Error due to different lengths for given variants and label! '
                    f'No. of higher subst. variants: {len(higher_variants)}. '
                    f'Number of given values: {len(higher_values)}.')

    # 1. CREATION OF LS AND TS SPLIT FOR SINGLE FOR LS AND HIGHER VARIANTS FOR TS
    all_variants = single_variants + higher_variants
    all_values = single_values + higher_values
    sub_ts = []  # Substitutions of TS
    values_ts = []  # Values of TS
    sub_ls = []
    values_ls = []

    if directed_evolution is False:
        if len(higher_variants) != 0:
            sub_ls = list(single_variants)  # Substitutions of LS
            values_ls = list(single_values)  # Values of LS
            for i in range(len(higher_variants)):
                if len(higher_variants) < 6:  # if less than 6 higher variants --> all higher variants to TS
                    sub_ts.append(higher_variants[i])
                    values_ts.append(higher_values[i])
                elif (i % 3) == 0 and i != 0:  # 1/4 of higher variants to TS, 3/4 to LS
                    sub_ts.append(higher_variants[i])
                    values_ts.append(higher_values[i])
                else:                       # 3/4 to LS
                    sub_ls.append(higher_variants[i])
                    values_ls.append(higher_values[i])
        else:  # if no higher substituted variants are available split 80%/20%
            random_nums = []
            range_list = np.arange(0, len(all_variants))
            while len(sub_ls) < len(all_variants) * 4 // 5:  # 80 % Learning Set
                random_num = random.choice(range_list)
                if random_num not in random_nums:
                    random_nums.append(random_num)
                    sub_ls.append(all_variants[random_num])
                    values_ls.append(all_values[random_num])
            else:  # 20 % Test Set
                for num in range_list:
                    if num not in random_nums:
                        sub_ts.append(all_variants[num])
                        values_ts.append(all_values[num])

    return sub_ls, values_ls, sub_ts, values_ts


def make_sub_ls_ts_randomly(
        single_variants,
        single_values,
        higher_variants,
        higher_values
):
    """
    Creation of learning set and test set by randomly splitting sets
    """
    length = len(single_variants) + len(higher_variants)
    range_list = np.arange(0, length)

    ts = []
    ls = []
    while len(ls) < length * 4 // 5:
        random_num = random.choice(range_list)
        if random_num not in ls:
            ls.append(random_num)

    for j in range_list:
        if j not in ls:
            ts.append(j)

    combined = single_variants + higher_variants  # substitutions
    combined2 = single_values + higher_values  # values

    sub_ls = []
    values_ls = []
    tot_sub_ls, tot_values_ls = [], []
    tot_sub_ts, tot_values_ts = [], []

    for i in ls:
        sub_ls.append(combined[i])
        values_ls.append(combined2[i])

    sub_ts = []
    values_ts = []
    for j in ts:
        sub_ts.append(combined[j])
        values_ts.append(combined2[j])

    for subs in sub_ls:
        for subs2 in sub_ts:
            if subs == subs2:
                logger.warning(f'\n<Warning> LS and TS overlap for: {subs} - '
                               f'You might want to consider checking the provided '
                               f'datasets for multiple entries')

    tot_sub_ls.append(sub_ls)
    tot_values_ls.append(values_ls)
    tot_sub_ts.append(sub_ts)
    tot_values_ts.append(values_ts)

    return tot_sub_ls[0], tot_values_ls[0], tot_sub_ts[0], tot_values_ts[0]


def make_fasta_ls_ts(
        filename,
        wt_seq,
        substitutions,
        fitness_values
):
    """
    Creates learning and test sets (.fasta style-like files with fitness values
    indicated by starting semicolon ';')

    filename: str
        String for defining the filename for the learning and test set "fasta-like" files.
    wt: str
        Wild-type sequence as string
    substitutions: list
        List of substitutions of a single variant of the format:
            - Single substitution variant, e.g. variant A123C: ['A123C']
            - Higher variants, e.g. variant A123C/D234E/F345G: ['A123C', 'D234E, 'F345G']
            --> Full substitutions list, e.g.: [['A123C'], ['A123C', 'D234E, 'F345G']]
    fitness_values: list
        List of ints/floats of the variant fitness values, e.g. for two variants: [1.4, 0.8]
    """
    myfile = open(filename, 'w')
    for i, var in enumerate(substitutions):  # var are lists of (single or multiple) substitutions
        temp = list(wt_seq)
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
        print(f'>{name}', file=myfile)
        print(f';{fitness_values[i]}', file=myfile)
        print(''.join(temp), file=myfile)
        # print(name+';'+str(val[i])+';'+''.join(temp), file=myfile)  # uncomment output: CSV format
    myfile.close()


def get_seqs_from_var_name(
        wt_seq,
        substitutions,
        fitness_values
) -> tuple[list, list, list]:
    """
    Similar to function above but just returns sequences

    wt: str
        Wild-type sequence as string
    substitutions: list
        List of substiutuions of a single variant of the format:
            - Single substitution variant, e.g. variant A123C: ['A123C']
            - Higher variants, e.g. variant A123C/D234E/F345G: ['A123C', 'D234E, 'F345G']
            --> Full substitutions list, e.g.: [['A123C'], ['A123C', 'D234E, 'F345G']]
    fitness_values: list
        List of ints/floats of the variant fitness values, e.g. for two variants: [1.4, 0.8]
    """
    variant, values, sequences = [], [], []
    for i, var in enumerate(substitutions):  # var are lists of (single or multiple) substitutions
        temp = list(wt_seq)
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
        variant.append(name)
        values.append(fitness_values[i])
        sequences.append(''.join(temp))

    return variant, values, sequences
