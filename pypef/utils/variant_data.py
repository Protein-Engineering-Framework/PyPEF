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

from __future__ import annotations
import os
import numpy as np
import pandas as pd


amino_acids = [
    'A', 'C', 'D', 'E', 'F',
    'G', 'H', 'I', 'K', 'L',
    'M', 'N', 'P', 'Q', 'R',
    'S', 'T', 'V', 'W', 'Y'
]


def get_wt_sequence(sequence_fasta):
    """
    Gets wild-type sequence from defined input file (can be pure sequence or fasta style)
    """
    if sequence_fasta is None:
        return None
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


def read_models(number):
    """
    reads the models found in the file Model_Results.txt.
    If no model was trained, the .txt file does not exist.
    """
    try:
        ls = ""
        with open('Model_Results.txt', 'r') as file:
            for i, lines in enumerate(file):
                if i == 0:
                    if lines[:6] == 'No FFT':
                        number += 2
                if i <= number + 1:
                    ls += lines
        return ls
    except FileNotFoundError:
        return "No Model_Results.txt found."


def absolute_path_cwd_file(file):
    """
    Get the current working directory
    """
    if file is None:
        return None
    return os.path.join(os.getcwd(), file)


def path_aaidx_txt_path_from_utils(filename):
    """
    returns the relative path to the /AAindex folder from the utils directory,
    e.g. path/to/pypef/utils/../aaidx/AAindex/FAUJ880104.txt.
    """
    modules_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(modules_path, '..', 'ml', 'AAindex', f'{filename}.txt')


def get_sequences_from_file(
        fasta: str,
        mult_path: str | None = None
) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    "Get_Sequences" reads (learning and test) .fasta and
    .fasta-like ".fasl" format files and extracts the name,
    the target value and the sequence of the protein.
    Only takes one-liner sequences for correct input.
    See example directory for required fasta file format.
    Make sure every marker (> and ;) is seperated by a
    space ' ' from the value respectively name.
    """
    if mult_path is not None:
        os.chdir(mult_path)

    sequences = []
    values = []
    names_of_mutations = []

    with open(fasta, 'r') as f:
        words = ""
        for line in f:
            if line.startswith('>'):
                if words != "":
                    sequences.append(words)
                words = line.split('>')
                names_of_mutations.append(words[1].strip())
                words = ""

            elif line.startswith('#'):
                pass  # are comments

            elif line.startswith(';'):
                if words != "":
                    sequences.append(words)
                words = line.split(';')
                values.append(float(words[1].strip()))
                words = ""

            else:
                try:
                    words += line.strip()
                except IndexError:
                    raise IndexError("Learning or Validation sets (.fasta) likely "
                                     "have emtpy lines (e.g. at end of file)")
        if words != "":
            sequences.append(words)
    # Check consistency
    if len(values) != 0:
        if len(sequences) != len(values):
            raise SystemError(
                f'Error: Number of sequences does not fit with number of target values! '
                f'Number of sequences: {str(len(sequences))}, Number of target values: {str(len(values))}.'
            )
    if mult_path is not None:
        os.chdir('..')

    return np.array(sequences), np.array(names_of_mutations), np.array(values)


def remove_nan_encoded_positions(
        xs: np.ndarray | list,
        *yss
):
    """
    Removes encoded sequence (x) of sequence list xs when NaNs occur in x.
    Also removes the corresponding fitness value y (f(x) --> y) at position i.
    ys can also be any type of list, e.g. variants or sequences.
    """
    xs = list(xs)
    temp = []
    for ys in yss:
        try:
            ys = list(np.atleast_1d(ys))
            if isinstance(ys, pd.Series):
                temp.append(list(ys))
            elif ys is None:
                if len(yss) == 1:
                    temp = (None,)
                else:
                    temp.append([None])
            else:
                if type(ys) == np.ndarray:
                    if np.array(ys).ndim == 0:
                        temp.append([list(np.atleast_1d(ys).tolist())])
                    else:
                        temp.append(list(np.atleast_1d(ys).tolist()))
                else:
                    temp.append(list(ys))
        except ValueError:
            temp.append(list(ys))
    if temp:
        yss = temp
    if not yss == () and not yss == (None,):
        for i, ys in enumerate(yss):
            assert len(xs) == len(ys), "Number of input sequences to be compared unequal."
            try:
                for j, x in enumerate(xs):
                    if np.shape(np.array(xs, dtype='object'))[1] and np.shape(np.array(ys, dtype='object'))[1]:
                        assert len(xs[j]) == len(ys[j]), "Length of input sequences to be compared unequal."
            except IndexError:
                break
    drop = []
    for i, x in enumerate(xs):
        try:
            if None in x:
                drop.append(i)
        except TypeError:
            raise TypeError(
                "Take lists of lists as input, e.g., for single sequence "
                "[[1, 2, 3, 4]]."
            )
    drop = sorted(drop, reverse=True)
    for idx in drop:
        del xs[idx]
        if not yss == () and not yss == (None,):
            for ys in yss:
                del ys[idx]
    if len(yss) == 1:
        return np.array(xs, dtype='object'), np.array(yss[0])

    return np.array(xs, dtype='object'), *np.array(yss, dtype='object')


def get_basename(filename: str) -> str:
    """
    Description
    -----------
    Extracts and returns the basename of the filename.

    Parameters
    ----------
    filename: str

    Returns
    -------
    str
        os.path.basename (filename) string without filename extension
    """
    return os.path.basename(filename).split('.')[0]


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


def split_variants(variants, sep='/'):
    """
    Splits variants according to mutation separator.
    """
    variants_splitted = []
    for variant in variants:
        variants_splitted.append(variant.split(sep))
    return variants_splitted


def read_csv(
        file_name: str,
        fitness_key: str = None
) -> tuple[list, list, list]:
    """
    Description
    -----------
    Reads input CSV file and return variants names and
    associated fitness values.

    Parameters
    ----------
    file_name: str
        Name of CSV file to read.
    fitness_key: str
        Name of column containing the fitness values.
        If None, column 1 (0-indexed) will be taken.

    Returns
    -------
    variants: np.ndarray
        Array of variant names
    fitnesses:
        Array of fitness values
    """
    df = pd.read_csv(file_name, sep=';', comment='#')
    if df.shape[1] == 1:
        df = pd.read_csv(file_name, sep=',', comment='#')
    if df.shape[1] == 1:
        df = pd.read_csv(file_name, sep='\t', comment='#')
    if fitness_key is not None:
        fitnesses = df[fitness_key].to_numpy(dtype=float)
    else:
        fitnesses = list(df.iloc[:, 1].to_numpy(dtype=float))
    variants = list(df.iloc[:, 0].to_numpy(dtype=str))
    features = list(df.iloc[:, 2:].to_numpy(dtype=float))

    return variants, fitnesses, features


def generate_dataframe_and_save_csv(
        variants: list,
        sequence_encodings: list,
        fitnesses: list,
        csv_file: str,
        encoding_type: str = '',
        save_df_as_csv: bool = True
) -> pd.DataFrame:
    """
    Description
    -----------
    Creates a pandas.DataFrame from the input data (numpy array including
    variant names, fitnesses, and encoded sequences).
    Writes pandas.DataFrame to a specified CSV file follwing the scheme:
    variants; fitness values; encoded sequences

    Parameters
    ----------
    variants: list
        Variant names.
    fitnesses: list
        Sequence-associated fitness value.
    sequence_encodings: list
        Sequence encodings (feature matrix) of sequences.
    csv_file : str
        Name of the csv file containing variant names and associated fitness values.
    encoding_type: str = ''
        Defines name for saved CSV file based on the chosen encoding technique:
        'aaidx', 'onehot', or 'dca'.
    save_df_as_csv : bool
        Writing DataFrame (Substitution;Fitness;Encoding_Features) to CSV (False/True).

    Returns
    -------
    df_dca: pandas.DataFrame
        Dataframe with variant names, fitness values, and features (encoded sequences).
        If save_df_as_csv is True also writes DF to CSV.
    """
    x = np.stack(sequence_encodings)
    feature_dict = {}            # Collecting features for each MSA position i
    for i in range(x.shape[1]):  # (encoding at pos. i) in a dict
        feature_dict[f'X{i + 1:d}'] = x[:, i]

    df_dca = pd.DataFrame()
    df_dca.insert(0, 'variant', variants)
    df_dca.insert(1, 'y', fitnesses)
    df_dca = pd.concat([df_dca, pd.DataFrame(feature_dict)], axis=1)

    if save_df_as_csv:
        filename = f'{get_basename(csv_file)}_{encoding_type}_encoded.csv'
        df_dca.to_csv(filename, sep=';', index=False)

    return df_dca


def process_df_encoding(df_encoding) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts the array of names, encoded sequences, and fitness values
    of the variants from the dataframe 'self.df_encoding'.
    It is mandatory that 'df_encoding' contains the names of the
    variants in the first column, the associated fitness value in the
    second column, and the encoded sequence starting from the third
    column.

    Returns
    -------
    Tuple of variant names, encoded sequences, and fitness values.
    """
    return (
        df_encoding.iloc[:, 0].to_numpy(),
        df_encoding.iloc[:, 2:].to_numpy(),
        df_encoding.iloc[:, 1].to_numpy()
    )


def read_csv_and_shift_pos_ints(
        infile: str,
        offset: int = 0,
        col_sep: str = ';',
        substitution_sep: str = '/',
        target_column: int = 1
):
    """
    Shifts position of substitutions of variants for all variants in the provided
    CSV file and saves the position-shifted variants with the corresponding fitness
    values to a new CSV file.
    """
    df = pd.read_csv(infile, sep=col_sep, comment='#')
    if df.shape[1] == 1:
        df = pd.read_csv(infile, sep=',', comment='#')
    if df.shape[1] == 1:
        df = pd.read_csv(infile, sep='\t', comment='#')
    try:
        df = df.dropna(subset=df.columns[[target_column]])  # if specific column has a NaN drop entire row
    except IndexError:
        raise IndexError("Did only detect a single column which might indicate a missing "
                         "target value column / a wrong specification of the CSV column "
                         "spearator character (e.g., --sep \';\').")

    column_1 = df.iloc[:, 0]
    column_2 = df.iloc[:, target_column].to_numpy()

    new_col = []

    for variant in column_1:
        if substitution_sep in variant:
            split_vars_list = []
            splitted_var = variant.split(substitution_sep)
            for s_var in splitted_var:
                new_var_int = int(s_var[1:-1]) - offset
                new_variant = s_var[0] + str(new_var_int) + s_var[-1]
                split_vars_list.append(new_variant)
            new_variant = ''
            for i, v in enumerate(split_vars_list):
                if i != len(split_vars_list) - 1:
                    new_variant += f'{v}/'  # substitution_sep replaced by '/'
                else:
                    new_variant += v
            new_col.append(new_variant)
        else:
            new_var_int = int(variant[1:-1]) - offset
            new_variant = variant[0] + str(new_var_int) + variant[-1]
            new_col.append(new_variant)

    data = np.array([new_col, column_2]).T
    new_df = pd.DataFrame(data, columns=['variant', 'fitness'])
    new_df.to_csv(infile[:-4] + '_new' + infile[-4:], sep=';', index=False)
