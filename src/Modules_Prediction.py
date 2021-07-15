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
Modules for making prediction files
from input CSV file (with value separators sep=',' or sep=';')
having the CSV-format

HEADER_VARIANTS;HEADER_VARIANTS_FITNESS
VARIANT_1;FITNESS_VALUE_1
VARIANT_2;FITNESS_VALUE_2
...

according to the self devised prediction set convention
> VARIANT_NAME_1
VARIANT_SEQUENCE_1
> VARIANT_NAME_2
VARIANT_SEQUENCE_2
...
"""


import os
import numpy as np
from tqdm import tqdm


def make_fasta_ps(filename, wt, substitution):
    """
    Creates prediction sets (.fasta style files)
    """
    myfile = open(filename, 'w')
    count = 0
    for i, var in enumerate(substitution):
        temporary = list(wt)
        name = ''
        separation = 0
        for single_var in var:
            position_index = int(str(single_var)[1:-1]) - 1
            new_amino_acid = str(single_var)[-1]
            temporary[position_index] = new_amino_acid
            if separation == 0:
                name += single_var
            else:
                name += '/' + single_var
            separation += 1
        print('>', name, file=myfile)
        print(''.join(temporary), file=myfile)
        count += 1
    myfile.close()


def make_combinations_double(arr):
    """
    Make double recombination variants
    """
    doubles = []
    for i in tqdm(range(len(arr))):
        for j in range(len(arr)):
            if j > i:
                if (arr[i][0])[1:-1] != (arr[j][0])[1:-1]:
                    doubles.append([arr[i][0], arr[j][0]])
                    if len(doubles) >= 8E04:
                        yield doubles
                        doubles = []
    yield doubles


def make_combinations_triple(arr):
    """
    Make triple recombination variants
    """
    length = len(arr)
    triples = []
    for i in tqdm(range(length)):
        for j in range(length):
            for k in range(length):
                if k > j > i:
                    if (arr[i][0])[1:-1] != (arr[j][0])[1:-1] != (arr[k][0])[1:-1]:
                        triples.append([arr[i][0], arr[j][0], arr[k][0]])
                        if len(triples) >= 8E04:
                            yield triples
                            triples = []
    yield triples


def make_combinations_quadruple(arr):
    """
    Make quadruple recombination variants
    """
    length = len(arr)
    quadruples = []
    for i in tqdm(range(length)):
        for j in range(length):
            for k in range(length):
                for l in range(length):
                    if l > k > j > i:
                        if (arr[i][0])[1:-1] != (arr[j][0])[1:-1] != (arr[k][0])[1:-1] != (arr[l][0])[1:-1]:
                            quadruples.append([arr[i][0], arr[j][0], arr[k][0], arr[l][0]])
                            if len(quadruples) >= 8E04:
                                yield quadruples
                                quadruples = []
    yield quadruples


def make_directory_and_enter(directory):
    """
    Makes directory for recombined or diverse prediction sets
    """
    previous_working_directory = os.getcwd()
    try:
        if not os.path.exists(os.path.dirname(directory)):
            os.mkdir(directory)
    except OSError:
        pass
    os.chdir(directory)

    return previous_working_directory


def create_split_files(array, single_variants, wt_sequence, name, no):
    """
    Creates split files from given variants for yielded recombined or diverse variants
    """
    if len(array) > 0:
        number_of_split_files = len(array) / (len(single_variants) * 20 ** 3)
        number_of_split_files = round(number_of_split_files)
        if number_of_split_files == 0:
            number_of_split_files += 1
        split = np.array_split(array, number_of_split_files)
        pwd = make_directory_and_enter(name + '_Split')
        for i in split:
            name_ = name + '_Split' + str(no) + '.fasta'
            make_fasta_ps(name_, wt_sequence, i)

        os.chdir(pwd)

        return ()


def make_combinations_double_all_diverse(arr, aminoacids):
    """
    Make double substituted naturally diverse variants
    """
    doubles = []
    for i in tqdm(range(len(arr))):
        for j in range(i + 1, len(arr)):
            for k in aminoacids:
                for l in aminoacids:
                    if ((arr[i][0])[1:-1]) != ((arr[j][0])[1:-1]) and\
                            ((arr[i][0])[:-1] + k)[0] != ((arr[i][0])[:-1] + k)[-1] and\
                            ((arr[j][0])[:-1] + l)[0] != ((arr[j][0])[:-1] + l)[-1]:
                        doubles.append(tuple([(arr[i][0])[:-1] + k, (arr[j][0])[:-1] + l]))  # tuple needed for
                        if len(doubles) >= 8E04:                                             # list(dict()):
                            doubles = list(dict.fromkeys(doubles))  # removes duplicated list entries
                            yield doubles
                            doubles = []
    doubles = list(dict.fromkeys(doubles))
    yield doubles


def make_combinations_triple_all_diverse(arr, aminoacids):
    """
    Make triple substituted naturally diverse variants
    """
    triples = []
    for i in tqdm(range(len(arr))):
        for j in range(i + 1, len(arr)):
            for k in range(j + 1, len(arr)):
                for l in aminoacids:
                    for m in aminoacids:
                        for n in aminoacids:
                            if ((arr[i][0])[1:-1]) != ((arr[j][0])[1:-1]) != ((arr[k][0])[1:-1]) and\
                                    ((arr[i][0])[:-1] + l)[0] != ((arr[i][0])[:-1] + l)[-1] and\
                                    ((arr[j][0])[:-1] + m)[0] != ((arr[j][0])[:-1] + m)[-1] and\
                                    ((arr[k][0])[:-1] + n)[0] != ((arr[k][0])[:-1] + n)[-1]:
                                triples.append(tuple([(arr[i][0])[:-1] + l, (arr[j][0])[:-1] + m,
                                                      (arr[k][0])[:-1] + n]))
                                if len(triples) >= 8E04:
                                    triples = list(dict.fromkeys(triples))  # transfer to dict and back to list
                                    yield triples
                                    triples = []
    triples = list(dict.fromkeys(triples))
    yield triples


def make_combinations_quadruple_all_diverse(arr, aminoacids):
    """
    Make quadruple substituted naturally diverse variants
    """
    quadruples = []
    for i in tqdm(range(len(arr))):
        for j in range(i + 1, len(arr)):
            for k in range(j + 1, len(arr)):
                for l in range(k + 1, len(arr)):
                    for m in aminoacids:
                        for n in aminoacids:
                            for o in aminoacids:
                                for p in aminoacids:
                                    if ((arr[i][0])[1:-1]) != ((arr[j][0])[1:-1]) != ((arr[k][0])[1:-1]) != \
                                            ((arr[l][0])[1:-1]) and\
                                            ((arr[i][0])[:-1] + m)[0] != ((arr[i][0])[:-1] + m)[-1] and\
                                            ((arr[j][0])[:-1] + n)[0] != ((arr[j][0])[:-1] + n)[-1] and\
                                            ((arr[k][0])[:-1] + o)[0] != ((arr[k][0])[:-1] + o)[-1] and\
                                            ((arr[l][0])[:-1] + p)[0] != ((arr[l][0])[:-1] + p)[-1]:
                                        quadruples.append(tuple([(arr[i][0])[:-1] + m, (arr[j][0])[:-1] + n,
                                                                 (arr[k][0])[:-1] + o, (arr[l][0])[:-1] + p]))
                                        if len(quadruples) >= 8E04:
                                            quadruples = list(dict.fromkeys(quadruples))  # transfer to dict
                                            yield quadruples                              # and back to list
                                            quadruples = []
    quadruples = list(dict.fromkeys(quadruples))
    yield quadruples
