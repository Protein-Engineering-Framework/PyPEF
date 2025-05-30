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
Contains Python code used for the approach presented in our 'hybrid modeling' paper
Preprint available at: https://doi.org/10.1101/2022.06.07.495081
Code available at: https://github.com/Protein-Engineering-Framework/Hybrid_Model

The included class 'CouplingsModel' has been taken from the script 'model.py' as part of the
EVmutation module (https://github.com/debbiemarkslab/EVmutation) written by Thomas Hopf in the
labs of Debora Marks and Chris Sander at Harvard Medical School and modified (shortened).
See also: https://doi.org/10.1038/nbt.3769
Hopf, T. A., Ingraham, J. B., Poelwijk, F.J., Schärfe, C.P.I., Springer, M., Sander, C., & Marks, D. S. (2016).
Mutation effects predicted from sequence co-variation. Nature Biotechnology, in press.

References:
[1] Hopf, T. A., Ingraham, J. B., Poelwijk, F.J., Schärfe, C.P.I., Springer, M., Sander, C., & Marks, D. S.
    Mutation effects predicted from sequence co-variation.
    Nature Biotechnology, 35, 2017, 128-135
    https://doi.org/10.1038/nbt.3769
[2] Hopf T. A., Green A. G., Schubert B., et al.
    The EVcouplings Python framework for coevolutionary sequence analysis.
    Bioinformatics 35, 2019, 1582-1584
    https://doi.org/10.1093/bioinformatics/bty862
[3] Ekeberg, M., Lövkvist, C., Lan, Y., Weigt, M., & Aurell, E.
    Improved contact prediction in proteins: Using pseudolikelihoods to infer Potts models.
    Physical Review E, 87(1), 2013, 012707. doi:10.1103/PhysRevE.87.012707
    https://doi.org/10.1103/PhysRevE.87.012707
"""

import logging
logger = logging.getLogger('pypef.dca.encoding')

import os
from collections.abc import Iterable
import numpy as np
from tqdm import tqdm
import pickle

from pypef.settings import USE_RAY
if USE_RAY:
    import ray

from pypef.utils.variant_data import amino_acids
from pypef.utils.helpers import ray_conditional_decorator

_SLICE = np.s_[:]


class InvalidVariantError(Exception):
    """
    Description
    -----------
    Exception raised when entered variant does not follow the required scheme
    (integer enclosed by two one-letter code representations of amino acids).

    Attributes
    ----------
    variant: str
        Variant that causes the error
    message: str
        Explanation of the error
    """

    def __init__(self, variant: str):
        self.variant = variant
        message = f"The entered variant '{self.variant}' does not follow the required scheme " \
                  f"(integer enclosed by two one letter code representations of amino acids). " \
                  f"Check separator or variant."
        self.message = message
        super().__init__(self.message)


class EffectiveSiteError(Exception):
    """
    Description
    -----------
    Exception raised when requested position is not implemented in the DCA model.

    Attributes
    ----------
    position: int
        Position that causes the error
    variant: str
        Variant including that position
    message: str
        Explanation of the error
    """

    def __init__(self, position: int, variant: str, verbose: bool = True):
        self.position = position
        self.variant = variant
        self.verbose = verbose
        message = f"The position {self.position} of variant '{self.variant}' is " \
                  f"not an effective site in the DCA model and thus cannot be predicted."
        if self.verbose:
            logger.info(message)
            self.message = message
            super().__init__(self.message)


def is_valid_substitution(substitution: str) -> bool:
    """
    Description
    -----------
    A substitution has to follow the scheme:
    First character: (wild-type/substituted) amino acid in one-letter code representation
    Last character: (introduced) amino acid in one-letter code representation
    In between: position (of substitution)

    If the entered substitution does not follow this scheme (integer enclosed by two one
    letter code representations of amino acids) return False, else return True.

    Parameters
    -----------
    substitution : str
        Substitution as string: Integer enclosed by two letters representing
        the wild-type (first) and variant amino acid (last) in one letter code.

    Returns
    -------
    boolian
    """
    if not substitution[0] in amino_acids:  # not accepting format IntegerAA, e.g., 123D
        return False

    if not substitution[-1] in amino_acids:
        return False

    try:
        int(substitution[1:-1])
    except ValueError:
        return False

    return True


def is_valid_variant(variant: str, separator='/') -> bool:
    """
    Description
    -----------
    Gets the single substitutions of the variant and checks if they follow the required scheme.

    If the entered substitution does not follow this scheme (integer enclosed by two one-
    letter code representations of amino acids) return False, else return True.

    Parameters
    ----------
    variant : str
        Joined string of integers enclosed by two letters representing the wild type
        and variant amino acid in the single letter code. -> Check separator
    separator : str
        Character to split the variant to obtain the single substitutions (default=',').

    Returns
    -------
    boolian
    """
    for substitution in variant.split(separator):
        if not is_valid_substitution(substitution):
            return False

    return True


def get_single_substitutions(variant: str, separator='/') -> Iterable:
    """
    Description
    -----------
    Generator that extracts and returns the single substitutions of the entered variant.

    Parameters
    ----------
    See 'is_valid_variant' for an explanation.

    Returns
    -------
    Generator object
    """
    if is_valid_variant(variant, separator):
        for substitution in variant.split(separator):
            yield substitution

    else:
        raise InvalidVariantError(variant)


class CouplingsModel:
    """
    Class to store parameters of pairwise undirected graphical model of sequences
    and compute evolutionary couplings, sequence statistical energies, etc.
    """
    def __init__(
            self,
            filename,
            precision="float32",
            verbose: bool = False,
            **kwargs
    ):
        """
        Initializes the object with raw values read from binary .Jij file

        Parameters
        ----------
        filename : str
            Binary Jij file containing model parameters from plmc software
        alphabet : str
            Symbols corresponding to model states (e.g. "-ACGT").
        precision : {"float32", "float64"}, default: "float32"
            Sets if input file has single (float32) or double precision (float64)
        """
        self.index_map = None
        self._target_seq = None
        self._index_list = None
        self.x_wt = None
        self.verbose = verbose
        try:
            self.__read_plmc_v2(filename, precision)
        except TypeError or FileNotFoundError:
            raise SystemError(
                "Did not find (specified) PLMC parameter file. "
                "The parameter file is required for DCA-based "
                "encoding and can be provided via the flag "
                "--params PLMC_FILE."
            )
        self.alphabet_map = {s: i for i, s in enumerate(self.alphabet)}

        # in non-gap mode, focus sequence is still coded with a gap character,
        # but gap is not part of model alphabet anymore; so if mapping crashes
        # that means there is a non-alphabet character in sequence array
        # and therefore there is no focus sequence.
        try:
            self.target_seq_mapped = np.array([self.alphabet_map[x] for x in self.target_seq])
            self.has_target_seq = (np.sum(self.target_seq_mapped) > 0)
        except KeyError:
            self.target_seq_mapped = np.zeros(shape=np.shape(self.l), dtype=np.int32)
            self.has_target_seq = False

    def __read_plmc_v2(self, filename, precision):
        """
        Read updated Jij file format from plmc.

        Parameters
        ----------
        filename : str
            Binary Jij file containing model parameters
        precision : {"float32", "float64"}
            Sets if input file has single or double precision

        """
        with open(filename, "rb") as f:
            # model length, number of symbols, valid/invalid sequences
            # and iterations
            self.l, self.num_symbols, self.n_valid, self.n_invalid, self.num_iter = (
                np.fromfile(f, "int32", 5)
            )

            # theta, regularization weights, and effective number of samples
            self.theta, self.lambda_h, self.lambda_j, self.lambda_group, self.n_eff = (
                np.fromfile(f, precision, 5)
            )

            # Read alphabet (make sure we get proper unicode rather than byte string)
            self.alphabet = np.fromfile(
                f, "S1", self.num_symbols
            ).astype("U1")

            # weights of individual sequences (after clustering)
            self.weights = np.fromfile(
                f, precision, self.n_valid + self.n_invalid
            )

            # target sequence and index mapping, again ensure unicode
            self._target_seq = np.fromfile(f, "S1", self.l).astype("U1")
            self.index_list = np.fromfile(f, "int32", self.l)

            # Analyzing Positions included in the PLMC file (based on the MSA)
            not_valid, valid = [], []
            for num in range(self.index_list[0], self.index_list[-1] + 1, 1):
                if num not in self.index_list:
                    not_valid.append(num)
                else:
                    valid.append(num)
            self.wt_aa_pos = []
            for aa, pos in zip(self._target_seq, self.index_list):
                self.wt_aa_pos.append(str(aa) + str(pos))
            if self.verbose:
                logger.info(f'Evaluating gap content of PLMC parameter file... '
                            f'First amino acid position used in the MSA (PLMC params file) is '
                            f'{self._target_seq[0]}{self.index_list[0]} and the last position '
                            f'used is {self._target_seq[-1]}{self.index_list[-1]}.')
                if len(not_valid) > 0:
                    logger.info(f'Further, non-included positions are:\n{str(not_valid)[1:-1]}')
                logger.info(f'Summary of all effective positions represented in the MSA '
                            f'based on wild-type sequence ({len(valid)} encoded positions):\n'
                            f'{str([aa_pos for aa_pos in self.wt_aa_pos])[1:-1]}'.replace("'", ""))

            # single site frequencies f_i and fields h_i
            self.f_i, = np.fromfile(
                f, dtype=(precision, (self.l, self.num_symbols)), count=1
            )

            self.h_i, = np.fromfile(
                f, dtype=(precision, (self.l, self.num_symbols)), count=1
            )

            # pair frequencies f_ij and pair couplings J_ij / J_ij
            self.f_ij = np.zeros(
                (self.l, self.l, self.num_symbols, self.num_symbols)
            )

            self.j_ij = np.zeros(
                (self.l, self.l, self.num_symbols, self.num_symbols)
            )

            for i in range(self.l - 1):
                for j in range(i + 1, self.l):
                    self.f_ij[i, j], = np.fromfile(
                        f, dtype=(precision, (self.num_symbols, self.num_symbols)),
                        count=1
                    )
                    self.f_ij[j, i] = self.f_ij[i, j].T

            for i in range(self.l - 1):
                for j in range(i + 1, self.l):
                    self.j_ij[i, j], = np.fromfile(
                        f, dtype=(precision, (self.num_symbols, self.num_symbols)),
                        count=1
                    )

                    self.j_ij[j, i] = self.j_ij[i, j].T

    def get_target_seq_and_index(self):
        """
        Gets and returns the target sequence of encodeable positions as
        well as the index list of encodeable positions that are the
        corresponding amino acid positions of the wild type sequence (1-indexed).

        Returns
        ----------
        self._target_seq: list
            List of single letter strings of the wild-type amino acids
            at the encodeable positions
        self._index_list: list
            List of integers of encodeable amino acid positions
        """
        return self._target_seq, self._index_list

    @property
    def target_seq(self):
        """
        Target/Focus sequence of model used for delta_hamiltonian
        calculations (including single and double mutation matrices)
        """
        return self._target_seq

    @target_seq.setter
    def target_seq(self, sequence):
        """
        Define a new target sequence

        Parameters
        ----------
        sequence : str, or list of chars
            Define a new default sequence for relative Hamiltonian
            calculations (e.g. energy difference relative to wild-type
            sequence).
            Length of sequence must correspond to model length (self.l)
        """
        if len(sequence) != self.l:
            raise ValueError(f"Sequence length inconsistent with model length: {len(sequence)} != {self.l}")

        if isinstance(sequence, str):
            sequence = list(sequence)

        self._target_seq = np.array(sequence)
        self.target_seq_mapped = np.array([self.alphabet_map[x] for x in self.target_seq])
        self.has_target_seq = True

    @property
    def index_list(self):
        """
        Target/Focus sequence of model used for delta_hamiltonian
        calculations (including single and double mutation matrices)
        """
        return self._index_list

    @index_list.setter
    def index_list(self, mapping):
        """
        Define a new number mapping for sequences

        Parameters
        ----------
        mapping: list of int
            Sequence indices of the positions in the model.
            Length of list must correspond to model length (self.l)
        """
        if len(mapping) != self.l:
            raise ValueError(f"Mapping length inconsistent with model length: "
                             f"{len(mapping)} != {self.l}\n"
                             f"Potentially the loaded PLMC parameter file is not "
                             f"a binary parameter file produced with PLMC.")

        self._index_list = np.array(mapping)
        self.index_map = {b: a for a, b in enumerate(self.index_list)}

    def __map(self, indices, mapping):
        """
        Applies a mapping either to a single index, or to a list of indices

        Parameters
        ----------
        indices : Iterable of items to be mapped, or single item
        mapping: Dictionary containing mapping into new space

        Returns
        -------
        Iterable, or single item
            Items mapped into new space
        """
        if ((isinstance(indices, Iterable) and not isinstance(indices, str)) or
                (isinstance(indices, str) and len(indices) > 1)):
            return np.array([mapping[i] for i in indices])
        else:
            return mapping[indices]

    def __4d_access(self, matrix, i=None, j=None, a_i=None, a_j=None):
        """
        Provides shortcut access to column pair properties
        (e.g. J_ij or f_ij matrices)

        Parameters
        -----------
        i : Iterable(int) or int
            Position(s) on first matrix axis
        j : Iterable(int) or int
            Position(s) on second matrix axis
        a_i : Iterable(str) or str
            Symbols corresponding to first matrix axis
        a_j : Iterable(str) or str
            Symbols corresponding to second matrix axis

        Returns
        -------
        np.array
            4D matrix "matrix" sliced according to values i, j, a_i and a_j
        """
        i = self.__map(i, self.index_map) if i is not None else _SLICE
        j = self.__map(j, self.index_map) if j is not None else _SLICE
        a_i = self.__map(a_i, self.alphabet_map) if a_i is not None else _SLICE
        a_j = self.__map(a_j, self.alphabet_map) if a_j is not None else _SLICE
        return matrix[i, j, a_i, a_j]

    def __2d_access(self, matrix, i=None, a_i=None):
        """
        Provides shortcut access to single-column properties
        (e.g. f_i or h_i matrices)

        Parameters
        -----------
        i : Iterable(int) or int
            Position(s) on first matrix axis
        a_i : Iterable(str) or str
            Symbols corresponding to first matrix axis

        Returns
        -------
        np.array
            2D matrix "matrix" sliced according to values i and a_i
        """
        i = self.__map(i, self.index_map) if i is not None else _SLICE
        a_i = self.__map(a_i, self.alphabet_map) if a_i is not None else _SLICE
        return matrix[i, a_i]

    def get_jij(self, i=None, j=None, a_i=None, a_j=None):
        """
        Quick access to J_ij matrix with automatic index mapping.
        See __4d_access for explanation of parameters.
        """
        return self.__4d_access(self.j_ij, i, j, a_i, a_j)

    def get_hi(self, i=None, a_i=None):
        """
        Quick access to h_i matrix with automatic index mapping.
        See __2d_access for explanation of parameters.
        """
        return self.__2d_access(self.h_i, i, a_i)


class PLMC(CouplingsModel):
    def __init__(
            self,
            params_file: str,
            separator: str = '/',
            verbose: bool = True
    ):
        """
        Class for performing the 'DCA encoding'.

        Attributes
        ----------
        params_file: str
            Binary parameter file outputted by PLMC.
        """
        super().__init__(filename=params_file)  # inherit functions and variables from class CouplingsModel
        self.verbose = verbose
        self.separator = separator
        target_seq, index = self.get_target_seq_and_index()
        self.x_wt = self.collect_encoded_sequences(target_seq[0] + str(index[0]) + target_seq[0])

    def _get_position_internal(self, position: int):
        """
        Description
        -----------
        Returns the "internal position" of an amino acid, e.g., D19V is the desired substitution,
        but the fasta sequence starts from residue 3, i.e., the first two residues are "missing".
        The DCA model will then recognize D19 as D17. In order to avoid wrong assignments,
        it is inevitable to calculate the "internal position" 'i'.

        Parameters
        ----------
        position : int
            Position of interest

        Returns
        -------
        i : int
            "Internal position" that may differ due to different starting residue.
        None
            If the requested position is not an active site.
        """
        offset = 0
        i = position - offset
        if i in self.index_list:
            return i
        else:
            return None

    def sum_ji(self, i: int, a_i: str, sequence: np.ndarray) -> float:
        """
        Description
        -----------
        Calculates the sum of all site-site interaction terms when site 'i' is occupied with amino acid 'a_i'.

        Parameters
        ----------
        i : int
            "Internal position" see '_get_position_internal' for an explanation.
        a_i : str
            Introduced amino acid at 'i' in one-letter code representation.
        sequence: np.ndarray
            Sequence of the variant as numpy array.

        Returns
        -------
        j_i : float
            Sum J(i) of all site-site interaction terms acting on position 'i' when occupied with 'a_i'.
        """
        j_i_sum = 0.0
        for j, a_j in zip(self.index_list, sequence):
            j_i_sum += self.get_jij(i=i, a_i=a_i, j=j, a_j=a_j)

        return j_i_sum

    @staticmethod
    def _unpack_substitution(substitution: str) -> tuple:
        """
        Description
        -----------
        Turns string representation of variant into tuple.

        Parameters
        ----------
        substitution : str
            Substitution as string: Integer enclosed by two letters representing
            the wild-type (first) and variant amino acid (last) in one letter code.

        Returns
        -------
        substitution : tuple
            (wild-type amino acid, position, variant amino acid)
        """
        return substitution[0], int(substitution[1:-1]), substitution[-1]

    def check_substitution_naming_against_wt(self, substitution: str, variant: str):
        """
        Checks whether the amino acid to substitute of the variant matches
        the amino acid of the wild type at this position.
        """
        if substitution[:-1] not in self.wt_aa_pos:
            wild_type_aa, position, a_i = self._unpack_substitution(substitution)
            raise SystemError(
                f"The variant naming scheme is not fitting to the PLMC "
                f"scheme. Substitution {substitution} of variant {variant} has "
                f"the amino acid {wild_type_aa} at position {position}, which "
                f"does not match the wild type sequence used as target for DCA-"
                f"based coupling parameter file generation. See summary of "
                f"(effective) wild-type positions and amino acids above. Please "
                f"check your input variant data or generate a new parameter file "
                f"for encoding."
            )

    def encode_variant(self, variant: str) -> np.ndarray:
        """
        Description
        -----------
        Encodes the variant using its "DCA representation".

        Parameters
        ----------
        variant : str
            Joined string of integers enclosed by two letters representing the wild type
            and variant amino acid in the single letter code. -> Check separator

        Returns
        -------
        x_var : np.ndarray
            Encoded sequence of the variant.
        """
        sequence = self.target_seq.copy()
        for substitution in get_single_substitutions(variant, self.separator):  # e.g. A123C/D234E --> A123C, D234C
            wild_type_aa, position, a_i = self._unpack_substitution(substitution)

            i = self._get_position_internal(position)
            if not i:
                raise EffectiveSiteError(position, variant, self.verbose)

            self.check_substitution_naming_against_wt(substitution, variant)
            i_mapped = self.index_map[i]
            sequence[i_mapped] = a_i

        x_var = np.zeros(sequence.size, dtype=float)
        for idx, (i, a_i) in enumerate(zip(self.index_list, sequence)):
            x_var[idx] = self.get_hi(i, a_i) + 0.5 * self.sum_ji(i, a_i, sequence)

        return x_var

    def collect_encoded_sequences(self, variants: list) -> np.ndarray:
        """
        Description
        -----------
        Collects all encoded sequences based on input variant names.

        Parameters
        ----------
        variants: list
            Variant name, e.g. 'A13E', or 'D127F' (wild-type sequence
            would be defined by substitution to itself, e.g. 'F17F').

        Returns
        ----------
        encoded_sequences: list
            List of all collected encoded sequences (features) for all
            inputted variants.
        non_encoded_variants_list_pos:list
            Internal array/list position for variants that could not be
            encoded due to the underlying MSA (inter-gap threshold for
            computing of local and coupling PLMC parameters). These list
            positions must then be removed for the corresponding fitness
            arrays/lists for training and testing the model.
        """
        encoded_sequences = []
        if len(np.atleast_1d(variants)) == 1:  # do not show progress bar for single variant
            set_silence = True  # thus, also not for directed evolution
        else:
            set_silence = False
        for i, variant in enumerate(tqdm(np.atleast_1d(variants), disable=set_silence)):
            try:
                encoded_sequences.append(self.encode_variant(variant))
            except EffectiveSiteError:
                encoded_sequences.append([None])

        return np.array(encoded_sequences, dtype='object')


"""
Below: Some helper functions to run the PLMC class and get 
the encoded sequences in parallel (threading) using Ray and
to construct a pandas.DataFrame to store the encoded sequences 
(features) and the associated fitness values in a CSV file.
"""


def save_plmc_dca_encoding_model(params_file, substitution_sep='/'):
    """
    Just converts plmc params from raw binary to
    Pickle-saved PLMC class.
    """
    logger.info("Transforming the provided plmc params file "
                "to a PLMC Pickle file (Pickles/PLMC).")
    plmc = PLMC(
        params_file=params_file,
        separator=substitution_sep,
        verbose=False
    )
    try:
        os.mkdir('Pickles')
    except FileExistsError:
        pass
    pickle.dump({
        'model': plmc,
        'model_type': 'PLMCpureDCA'
    },
        open(f'Pickles/PLMC', 'wb')
    )


def get_encoded_sequence(
        variant: str,
        dca_encode: PLMC
):
    """
    Description
    -----------
    Gets encoded sequence based on input variant name and a preexisting
    PLMC instance.

    Parameters
    ----------
    variant: str
        Variant name, e.g. 'A13E', or 'D127F'. Wild-type sequence
        is defined by substitution to itself, e.g. 'F17F'.
    dca_encode: PLMC class object
        For encoding sequences, see above: class PLMC.
    """
    try:
        encoded_seq = dca_encode.encode_variant(variant)
    except EffectiveSiteError:  # position not included in processed MSA
        return

    return encoded_seq


@ray_conditional_decorator
def _get_data_parallel(
        variants: list,
        sequences: list,
        fitnesses: list,
        dca_encode: PLMC,
        data: list
) -> list:
    """
    Description
    -----------
    Get the variant name, the associated fitness value, and its ("DCA"-)encoded sequence.

    Parameters
    ----------
    variants : list
        List of strings containing the variants to be encoded.
    fitnesses : list
        List of floats (1d) containing the fitness values associated to the variants.
    dca_encode : object
        Initialized 'PLMC' class object.
    data : manager.list()
        Manager.list() object to store the output of multiple processors.

    Returns
    -------
    data : manager.list()
        Filled list with variant names, fitnesses, and encoded sequence.
    """
    for i, (variant, sequence, fitness) in enumerate(zip(variants, sequences, fitnesses)):
        try:
            data.append([variant, sequence, dca_encode.encode_variant(variant), fitness])
        except EffectiveSiteError:  # do not append non-encoded sequences and
            pass                    # associated fitness values

    return data


def get_dca_data_parallel(
        variants: list,
        sequences: list,
        fitnesses: list,
        dca_encode: PLMC,
        threads: int,
        verbose=True
) -> tuple[list, list, list, list]:
    """
    Description
    -----------
    This function allows to generate the encoded sequences based on the variants
    given in 'csv_file' in a parallel manner.

    Parameters
    ----------
    variants: list (or np.ndarray)
        Variant names.
    sequences: list (or np.ndarray)
        Variant-associated protein sequences.
    fitnesses: list (or np.ndarray)
        Variant-associated fitness values.
    dca_encode : object
        Initialized 'PLMC' class object.
    threads : int
        Number of processes to be used for parallel execution.
        n_cores = 1 defines no threading (not using Ray).
    verbose: bool
        Logging message on/off.

    Returns
    -------
    data: numpy.ndarray
        Filled numpy array including variant names, fitnesses, and encoded sequences.
    non_effective_subs: list
        List of variant names that cannot be used for modelling as they are not effective
        positions in the underlying MSA used for generating local and coupling terms.
    """
    if verbose:
        logger.info(f'{len(variants)} input variants. Encoding variant sequences using parameters '
                    f'taken from plmc generated file. This might take some time...')

    idxs_nan = np.array([i for i, b in enumerate(np.isnan(fitnesses)) if b])  # find fitness NaNs
    if idxs_nan.size > 0:  # remove NaNs if present
        logger.info(f'Fitness NaNs are: {idxs_nan}')
        fitnesses = np.delete(fitnesses, idxs_nan)
        variants = np.delete(variants, idxs_nan)

    variants_split = np.array_split(variants, threads)    # split array in n_cores parts
    sequences_split = np.array_split(sequences, threads)  # for Ray parallelization
    fitnesses_split = np.array_split(fitnesses, threads)
    results = ray.get([
        _get_data_parallel.remote(
            variants_split[i],
            sequences_split[i],
            fitnesses_split[i],
            dca_encode,
            []
        ) for i in range(len(variants_split))
    ])

    data = [item for sublist in results for item in sublist]  # fusing all the individual results
    variants = [item[0] for item in data]
    sequences = [item[1] for item in data]
    xs = [item[2] for item in data]
    fitnesses = [item[3] for item in data]

    if verbose:
        logger.info(f'{len(data)} variants after NaN-valued and non-effective '
                    f'site-substituted variant (EffectiveSiteError) dropping.')

    return variants, sequences, xs, fitnesses
