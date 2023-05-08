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


def make_fasta_ps(
        filename,
        wt,
        substitution
):
    """
    Creates prediction sets (.fasta style files, i.e. without fitness values)
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


def make_recombinations_double(arr: tuple) -> list:
    """
    Description
    -----------
    Make double recombinant variants.

    Parameters
    ----------
    arr : tuple
        Lists if single substitutions in tuple, e.g.,
        (['L215F'], ['A217N'], ['R219S'], ['L249Y'])

    Returns
    -------
    doubles : list
        List of double substitution lists, e.g.,
        [['L215F', 'A217N'], ['L215F', 'R219S'], ['L215F', 'L249Y'],
        ['A217N', 'R219S'], ['A217N', 'L249Y'], ['R219S', 'L249Y']]
    """
    doubles = []
    arr_pos = [int(substitution[0][1:-1]) for substitution in arr]
    arr_pos, arr = zip(*sorted(zip(arr_pos, arr), key=lambda x: x[0]))
    for i in tqdm(range(len(arr))):
        for j in range(len(arr)):
            if j > i:
                if (arr[i][0])[1:-1] != (arr[j][0])[1:-1]:
                    doubles.append([arr[i][0], arr[j][0]])
                    if len(doubles) >= 8E04:
                        yield doubles
                        doubles = []
    yield doubles


def make_recombinations_triple(arr: list):
    """
    Description
    -----------
    Make triple recombinant variants.

    Parameters
    ----------
    arr: list
        List of single substitutions in tuple, e.g.,
        (['L215F'], ['A217N'], ['R219S'], ['L249Y'])

    Returns
    -------
    triples: list
        List of triple substitution lists, e.g.,
        [['L215F', 'A217N', 'R219S'], ['L215F', 'A217N', 'L249Y'],
        ['L215F', 'R219S', 'L249Y'], ['A217N', 'R219S', 'L249Y']]
    """
    length = len(arr)
    arr_pos = [int(substitution[0][1:-1]) for substitution in arr]
    arr_pos, arr = zip(*sorted(zip(arr_pos, arr), key=lambda x: x[0]))
    triples = []
    for i in tqdm(range(length)):
        for j in range(length):
            for k in range(length):
                if k > j > i:
                    if (arr[i][0])[1:-1] != (arr[j][0])[1:-1] and \
                            (arr[i][0])[1:-1] != (arr[k][0])[1:-1] and \
                            (arr[j][0])[1:-1] != (arr[k][0])[1:-1]:
                        triples.append([arr[i][0], arr[j][0], arr[k][0]])
                        if len(triples) >= 8E04:
                            yield triples
                            triples = []
    yield triples


def make_recombinations_quadruple(arr):
    """
    Description
    -----------
    Make quadruple recombination variants.

    Parameters
    ----------
    arr: list
        List of single substitutions in tuple, e.g.,
        (['L215F'], ['A217N'], ['R219S'], ['L249Y'])

    Returns
    -------
    quadruples: list
        List of quadruple substitution lists, e.g.,
        [['L215F', 'A217N', 'R219S', 'L249Y']]
    """
    length = len(arr)
    arr_pos = [int(substitution[0][1:-1]) for substitution in arr]
    arr_pos, arr = zip(*sorted(zip(arr_pos, arr), key=lambda x: x[0]))
    quadruples = []
    for i in tqdm(range(length)):
        for j in range(length):
            for k in range(length):
                for l in range(length):
                    if l > k > j > i:
                        if (arr[i][0])[1:-1] != (arr[j][0])[1:-1] and \
                                (arr[i][0])[1:-1] != (arr[k][0])[1:-1] and \
                                (arr[i][0])[1:-1] != (arr[l][0])[1:-1] and \
                                (arr[j][0])[1:-1] != (arr[k][0])[1:-1] and \
                                (arr[j][0])[1:-1] != (arr[l][0])[1:-1] and \
                                (arr[k][0])[1:-1] != (arr[l][0])[1:-1]:
                            quadruples.append([arr[i][0], arr[j][0], arr[k][0], arr[l][0]])
                            if len(quadruples) >= 8E04:
                                yield quadruples
                                quadruples = []
    yield quadruples


def make_recombinations_quintuple(arr):
    """
    Make quintuple recombination variants.

    :parameter arr: List(s) of all available single substitution(s)
    in tuple, e.g.,
    (['L215F'], ['A217N'], ['R219S'], ['L249Y'], ['P252I'])

    :returns quintuples: List of quintuple substitution lists, e.g.,
    [['L215F', 'A217N', 'R219S', 'L249Y', 'P252I']]
    """
    length = len(arr)
    arr_pos = [int(substitution[0][1:-1]) for substitution in arr]
    arr_pos, arr = zip(*sorted(zip(arr_pos, arr), key=lambda x: x[0]))
    quintuples = []
    for i in tqdm(range(length)):
        for j in range(length):
            for k in range(length):
                for l in range(length):
                    for m in range(length):
                        if m > l > k > j > i:
                            if (arr[i][0])[1:-1] != (arr[j][0])[1:-1] and \
                                    (arr[i][0])[1:-1] != (arr[k][0])[1:-1] and \
                                    (arr[i][0])[1:-1] != (arr[l][0])[1:-1] and \
                                    (arr[i][0])[1:-1] != (arr[m][0])[1:-1] and \
                                    (arr[j][0])[1:-1] != (arr[k][0])[1:-1] and \
                                    (arr[j][0])[1:-1] != (arr[l][0])[1:-1] and \
                                    (arr[j][0])[1:-1] != (arr[m][0])[1:-1] and \
                                    (arr[k][0])[1:-1] != (arr[l][0])[1:-1] and \
                                    (arr[k][0])[1:-1] != (arr[m][0])[1:-1] and \
                                    (arr[l][0])[1:-1] != (arr[m][0])[1:-1]:
                                quintuples.append([arr[i][0], arr[j][0], arr[k][0], arr[l][0], arr[m][0]])
                                if len(quintuples) >= 8E04:
                                    yield quintuples
                                    quintuples = []
    yield quintuples


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


def create_split_files(
        array,
        single_variants,
        wt_sequence,
        name,
        no
):
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

    :parameter arr: List of single substitutions in tuple, e.g.,
    (['L215F'], ['A217N'], ['R219S'], ['L249Y'])
    :parameter aminoacids: List of amino acids to combine, e.g., all 20 naturally occuring:
    ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
    'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

    :returns doubles: List of double (fully) diverse substitution tuples, e.g.,
    [('L215A', 'A217C'), ('L215A', 'A217D'), ('L215A', 'A217E'), ('L215A', 'A217F'), ...,
    ('R219Y', 'L249T'), ('R219Y', 'L249V'), ('R219Y', 'L249W'), ('R219Y', 'L249Y')]
    """
    doubles = []
    for i in tqdm(range(len(arr))):
        for j in range(i + 1, len(arr)):
            for k in aminoacids:
                for l in aminoacids:
                    """
                    Make sure that following substitution types are not
                    included for prediction. Examples:
                    1. Both simultaneous substitutions define exactly the 
                      same substitution at the same position, e.g., A1C/A1C:
                        (arr[i][0])[1:-1] != (arr[j][0])[1:-1]
                    2. "To-Wild-Type-Substitutions" at a single position e.g., A1A:
                        ((arr[i][0])[:-1] + k)[0] != ((arr[i][0])[:-1] + k)[-1]  # e.g., A1A
                        ((arr[j][0])[:-1] + l)[0] != ((arr[j][0])[:-1] + l)[-1]  # e.g., C2C
                    3. Just reversed substitution patterns, e.g., A1C/A2D and A2D/A1C
                      in doubles tuple (only possible until results not emptied/yielded 
                      and should generally not occur often):
                        not tuple([(arr[j][0])[:-1] + l, (arr[i][0])[:-1] + k]) in doubles  
                    """
                    if (arr[i][0])[1:-1] != (arr[j][0])[1:-1] and \
                            ((arr[i][0])[:-1] + k)[0] != ((arr[i][0])[:-1] + k)[-1] and \
                            ((arr[j][0])[:-1] + l)[0] != ((arr[j][0])[:-1] + l)[-1] and \
                            not tuple([(arr[j][0])[:-1] + l, (arr[i][0])[:-1] + k]) in doubles:
                        doubles.append(tuple([(arr[i][0])[:-1] + k, (arr[j][0])[:-1] + l]))  # tuple needed for
                        if len(doubles) >= 8E04:                                             # list(dict()):
                            doubles = list(dict.fromkeys(doubles))  # removes duplicated list entries
                            yield doubles
                            doubles = []
    doubles = list(dict.fromkeys(doubles))
    yield doubles


def make_combinations_double_all_diverse_and_all_positions(wt_seq, aminoacids):
    """
    Make double substituted naturally diverse variants

    :parameter arr: List of single substitutions in tuple, e.g.,
    (['L215F'], ['A217N'], ['R219S'], ['L249Y'])
    :parameter aminoacids: List of amino acids to combine, e.g., all 20 naturally occuring:
    ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
    'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

    :returns doubles: List of double (fully) diverse substitution lists, e.g.,
    [('L215A', 'A217C'), ('L215A', 'A217D'), ('L215A', 'A217E'), ('L215A', 'A217F'), ...,
    ('R219Y', 'L249T'), ('R219Y', 'L249V'), ('R219Y', 'L249W'), ('R219Y', 'L249Y')]
    """
    counter = 0
    doubles = []
    for i in tqdm(range(len(wt_seq))):
        for j in range(i + 1, len(wt_seq)):
            for k in aminoacids:
                pos_1 = wt_seq[i] + str(i + 1) + str(k)
                for l in aminoacids:
                    pos_2 = wt_seq[j] + str(j + 1) + str(l)
                    if pos_1[0] != pos_1[-1] \
                            and pos_2[0] != pos_2[-1] \
                            and pos_1[1:-1] != pos_2[1:-1]:
                        doubles.append(tuple([pos_1, pos_2]))  # tuple needed for
                        if len(doubles) >= 8E04:                    # list(dict()):
                            doubles = list(dict.fromkeys(doubles))  # removes duplicated list entries
                            counter += len(doubles)
                            yield doubles
                            doubles = []
    doubles = list(dict.fromkeys(doubles))
    yield doubles


def make_combinations_triple_all_diverse(arr, aminoacids):
    """
    Make triple substituted naturally diverse variants.
    Analogous to function "make_combinations_double_all_diverse"
    but yielding three combined substitutions.
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
    Make quadruple substituted naturally diverse variants.
    Analogous to function "make_combinations_double_all_diverse"
    but yielding four combined substitutions.
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
                                    if ((arr[i][0])[1:-1]) \
                                            != ((arr[j][0])[1:-1]) \
                                            != ((arr[k][0])[1:-1]) \
                                            != ((arr[l][0])[1:-1]) \
                                            and\
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


if __name__ == '__main__':
    k = list(make_recombinations_quintuple((
        ['A86V'], ['T91S'], ['M108Q'], ['A109E'], ['T111P'], ['A86S'], ['T91E'], ['M108L'], ['A109S'], ['T111G'],
        ['M108R'], ['T111N'], ['T91V'], ['M108T'], ['A109G'], ['T111F'], ['T91A'], ['A109M'], ['A86D'], ['T91R'],
        ['A109K'], ['T111D'], ['T91Q'], ['A109V'], ['T111S'], ['A86C'], ['T91L'], ['A109T'], ['M108S'], ['A109F'],
        ['T111L'], ['A86T'], ['A109Q'], ['M108A'], ['A109P'], ['T111Q'], ['A86N'], ['T91Y'], ['A109L'], ['T111A'],
        ['T91F'], ['A109Y'], ['A86I'], ['A109D'], ['M108K'], ['M108I'], ['T91N'], ['T111C'], ['T91M'], ['T91C'],
        ['M108P'], ['T111M'], ['T91H'], ['M108C'], ['M108F'], ['M108G'], ['A109N'], ['M108E'], ['A109W'], ['M108W'],
        ['A109I'], ['T91P'], ['M108H'], ['T91D'], ['A109R'], ['T91I'], ['M108Y'], ['T91G'], ['T91W'], ['A86R'],
        ['T91K'], ['T111Y'], ['M108D'], ['A86W'], ['M108V'], ['T111I'], ['M108N'], ['A109C'], ['A109H']
    )))
    for i, k_ in enumerate(k):
        print(i + 1, np.shape(k_))
    # (10 * 80,000 (* 5)) + (1 * 2503 (* 5))
