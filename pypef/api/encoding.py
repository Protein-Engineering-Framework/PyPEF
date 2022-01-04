#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on 05 October 2020
# @author: Niklas Siedhoff, Alexander-Maurice Illig
# <n.siedhoff@biotec.rwth-aachen.de>, <a.illig@biotec.rwth-aachen.de>
# PyPEF - Pythonic Protein Engineering Framework
# Released under Creative Commons Attribution-NonCommercial 4.0 International Public License (CC BY-NC 4.0)
# For more information about the license see https://creativecommons.org/licenses/by-nc/4.0/legalcode

# PyPEF – an Integrated Framework for Data-driven Protein Engineering
# Niklas E. Siedhoff1,§, Alexander-Maurice Illig1,§, Ulrich Schwaneberg1,2, Mehdi D. Davari1,*
# 1Institute of Biotechnology, RWTH Aachen University, Worringer Weg 3, 52074 Aachen, Germany
# 2DWI-Leibniz Institute for Interactive Materials, Forckenbeckstraße 50, 52074 Aachen, Germany
# *Corresponding author
# §Equal contribution

import os
import numpy as np
import re


def full_path(filename):
    """
    returns the path of an index inside the folder /AAindex/,
    e.g. path/to/AAindex/FAUJ880109.txt.
    """
    modules_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(modules_path, '../AAindex/' + filename)


def path_aaindex_dir():
    """
    returns the path to the /AAindex folder, e.g. path/to/AAindex/.
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), '../AAindex')


def aa_index_list():
    return [full_path(file) for file in os.listdir(path_aaindex_dir()) if file.endswith('.txt')]


def get_variant_seqs(wild_type_sequence, sub):  # sub = list of single/higher substitution strings
    """
    Requires WT sequence, and substitution names.
    Returns variant sequences as list.

    Gets sequences from substitution names, i.e.,
    for WT sequence = 'ACDE' it returns for substitution 'D3F'
    the sequence 'ACFE'.
    """
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    names, sequences, subs_list = [], [], []
    for i, variant in enumerate(sub):  # var are strings of (single or multiple ('/')) substitutions
        if '/' in variant:  # higher substituted variants
            m = re.split(r'/', variant)
            for a, splits in enumerate(m):
                if splits[0].isdigit() or splits[0] in amino_acids and splits[-1] in amino_acids:
                    new = int(re.findall(r'\d+', splits)[0])
                    if splits[0] in amino_acids:
                        if splits[0] != wild_type_sequence[new - 1]:
                            raise NameError('Position of amino acids in given sequence does not match the given '
                                            'positions in the input data! E.g. see position {} and position {} '
                                            'being {} in the given sequence'.format(
                                variant, new, wild_type_sequence[new - 1])
                            )
                    higher_var = wild_type_sequence[new - 1] + str(new) + str(splits[-1])
                    m[a] = higher_var
                    if a == len(m) - 1:
                        subs_list.append(m)
        else:  # single substituted variants
            if variant.upper() == 'WT':
                subs_list.append([variant])
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
                                            ' in the given sequence.'.format(variant, num, wt[num - 1]))
                    except IndexError:
                        raise IndexError("Found variant sequence position {} in data which "
                                         "is out of range of wild-type sequence length.".format(str(num)))
                try:
                    full_variant = wild_type_sequence[num - 1] + str(num) + str(variant[-1])
                except IndexError:
                    raise IndexError("Found variant sequence position {} in data which "
                                     "is out of range of wild-type sequence length.".format(str(num)))
                subs_list.append([full_variant])

    for i, var in enumerate(subs_list):
        temp = list(wild_type_sequence)
        name = ''
        separation = 0

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
        sequences.append(''.join(temp))

    return sequences



class AAIndexDict:
    """
    gets all the information that are given in each AAindex file.
    For the program routine it provides the library to enable translation
    of the alphabetical amino acid sequence to an array of numericals.
    """
    def __init__(self, filename):
        self.File = filename
        self.Accession_Number = None
        self.Data_Description = None
        self.PMID = None
        self.Authors = None
        self.Title_Of_Article = None
        self.Journal_Reference = None

    def general_information(self):
        """
        Gets and allocates general information based on the AAindex file
        format defined by file sections 'H', 'D', 'E', 'A', 'T', 'J'
        """
        with open(self.File, 'r') as f:
            for line in f:
                # try/ except "removes" empty lines.
                try:
                    words = line.split()
                    id_letter = words[0]
                except IndexError:
                    break

                # Extract some general information about AAindex file.
                if id_letter == 'H':
                    self.Accession_Number = words[1]
                elif id_letter == 'D':
                    self.Data_Description = words[1]
                elif id_letter == 'E':
                    self.PMID = words[1:]
                elif id_letter == 'A':
                    self.Authors = ' '.join(words[1:])
                elif id_letter == 'T':
                    self.Title_Of_Article = ' '.join(words[1:])
                elif id_letter == 'J':
                    self.Journal_Reference = ' '.join(words[1:])

    def encoding_dictionary(self):
        """
        Get numerical values of AAindex for each amino acid
        """
        with open(self.File, 'r') as f:
            for line in f:
                # try/ except "removes" empty lines
                try:
                    words = line.split()
                    id_letter = words[0]
                except IndexError:
                    break

                # Extract numerical values of AAindex.
                if id_letter == 'I':

                    keys = []
                    for word in words[1:]:
                        keys.append(word[0])
                        keys.append(word[-1])

                    values = []
                    for row in range(2):
                        line = f.readline()
                        strings = line.split()
                        for idx, string in enumerate(strings):
                            # Some amino acids may have no value
                            try:
                                strings[idx] = float(string)
                            except ValueError:
                                strings[idx] = None
                        values.append(strings)
                    values = np.reshape(np.array(values).T, len(keys))

                    return dict(zip(keys, values))


class AAIndexEncoding:  # was class XY originally
    """
    converts the string sequence into a list of numericals using the AAindex translation library,
    Fourier transforming the numerical array that was translated by get_numerical_sequence --> do_fourier,
    computing the input matrices X and Y for the PLS regressor (get_x_and_y). Returns FFT-ed arrays (x),
    labels array (y), and raw_encoded sequences arrays (raw_numerical_sequences)
    """
    def __init__(self, aaindex_file, sequences):  # , value, mult_path=None, prediction=False):
        aaidx = AAIndexDict(aaindex_file)
        self.dictionary = aaidx.encoding_dictionary()
        # self.name, self.value = name, value  #ADD?
        self.sequences = sequences

    def get_numerical_sequence(self, sequence):
        return np.array([self.dictionary[aminoacid] for aminoacid in sequence])

    @staticmethod
    def do_fourier(sequence):
        """
        This static function does the Fast Fourier Transform. Since the condition

                    len(Array) = 2^k -> k = log_2(len(Array))
                    k in N

        must be satisfied, the array must be reshaped (zero padding) if k is no integer value.
        The verbose parameter prints also the real and imaginary part separately.
        """
        threshold = 1e-8  # errors due to computer uncertainties
        k = np.log2(sequence.size)  # get exponent k
        mean = np.mean(sequence, axis=0)  # calculate mean of numerical array
        sequence = np.subtract(sequence, mean)  # subtract mean to avoid artificial effects of FT

        if abs(int(k) - k) > threshold:  # check if length of array fulfills previous equation
            numerical_sequence_reshaped = np.zeros(pow(2, (int(k) + 1)))  # reshape array
            for index, value in enumerate(sequence):
                numerical_sequence_reshaped[index] = value
            sequence = numerical_sequence_reshaped

        fourier_transformed = np.fft.fft(sequence)  # FFT
        ft_real = np.real(fourier_transformed)
        ft_imag = np.imag(fourier_transformed)

        x = np.linspace(1, sequence.size, sequence.size)  # frequencies
        x = x / max(x)  # normalization of frequency

        amplitude = ft_real * ft_real + ft_imag * ft_imag

        if max(amplitude) != 0:
            amplitude = np.true_divide(amplitude, max(amplitude))  # normalization of amplitude

        return amplitude, x

    def get_x_and_y(self):
        """
        getting the input matrices X (FFT amplitudes) and Y (variant labels)
        """
        frequencies = []
        amplitudes = []
        raw_numerical_seq = []

        for sequence in self.sequences:
            num = self.get_numerical_sequence(sequence)

            # There may be amino acids without a value in AAindex.
            # Skip these Indices.
            if None in num:
                break

            # Numerical sequence gets expended by zeros so that also different lengths of sequences
            # can be processed using '--nofft' option
            k = np.log2(len(num))
            if abs(int(k) - k) > 1e-8:  # check if length of array fulfills previous equation
                num_appended = np.append(num, np.zeros(pow(2, (int(k) + 1)) - len(num)))  # reshape array
            else:
                num_appended = num

            amplitudes_, frequencies_ = self.do_fourier(num)

            # Fourier spectra are mirrored at frequency = 0.5. No more information at higher frequencies.
            half = len(frequencies_) // 2  # // for integer division
            frequencies.append(frequencies_[:half])
            amplitudes.append(amplitudes_[:half])    # FFT-ed encoded amino acid sequences
            raw_numerical_seq.append(num_appended)  # Raw encoded amino acid sequences

        amplitudes = np.array(amplitudes)
        # frequencies = np.array(frequencies)  # not used
        raw_numerical_seq = np.array(raw_numerical_seq)

        x = amplitudes

        return x, raw_numerical_seq
