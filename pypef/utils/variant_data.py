# PyPEF - Pythonic Protein Engineering Framework
# https://github.com/niklases/PyPEF

from __future__ import annotations
import collections
import os
import numpy as np
import pandas as pd
from Bio import Align
from Bio.PDB import PDBParser
from sklearn.model_selection import GroupShuffleSplit

import logging
logger = logging.getLogger('pypef.utils.variant_data')

amino_acids = [
    'A', 'C', 'D', 'E', 'F',
    'G', 'H', 'I', 'K', 'L',
    'M', 'N', 'P', 'Q', 'R',
    'S', 'T', 'V', 'W', 'Y'
]


def get_wt_sequence(sequence_fasta):
    """
    Gets wild-type sequence from defined input file (can be pure sequence or fasta style).
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
    Reads the models found in the file Model_Results.txt.
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
    Gets the current working directory.
    """
    if file is None:
        return None
    return os.path.join(os.getcwd(), file)


def path_aaidx_txt_path_from_utils(filename):
    """
    Returns the relative path to the /AAindex folder from the utils directory,
    e.g. path/to/pypef/utils/../aaidx/AAindex/FAUJ880104.txt.
    """
    modules_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(modules_path, '..', 'ml', 'AAindex', f'{filename}.txt')


def get_sequences_from_file(
        fasta: str,
        mult_path: str | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads (learning and test) .fasta and .fasta-like ".fasl" 
    format files and extracts the name, the target value and 
    the sequence of the protein.
    Only takes one-liner sequences for correct input.
    Make sure every marker (> and ;) is seperated by a
    space ' ' from the value respectively name.

    Returns
    ----------
    np.array(sequences), np.array(names_of_mutations), np.array(values)
    """
    if mult_path is not None:
        os.chdir(mult_path)

    sequences = []
    values = []
    names_of_mutations = []

    allowed_chars = "ABCDEFGHIKLMNPQRSTVWYX-."
    allowed_chars += allowed_chars.lower()

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
                    line = line.strip()
                    if any(not c in line for c in allowed_chars):
                        for c in line:
                            if c not in allowed_chars:
                                pass
                                #raise SystemError(
                                #    f"The input file(s) (MSA or train/test sets) contain(s) unknown protein sequence characters "
                                #    f"(e.g.: \"{c}\"). Note that an MSA has to be provided in FASTA or A2M format (or formatted as "
                                #    F"pure linebreak-separated sequences).")
                    words += line
                except IndexError:
                    raise IndexError("Sequences in input file(s) likely "
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

    return np.array(sequences), np.array(names_of_mutations), np.asarray(values, dtype=np.float64)


def get_seqs_from_var_name(
        wt_seq: str,
        substitutions: list,
        fitness_values: None | list = None,
        shift_pos: int = 0
) -> tuple[list, list, list]:
    """
    Similar to function "get_sequences_from_file" but instead of getting 
    sequences from fasta file it directly gets them from wt sequence and
    variant specifiers.

    wt_seq: str
        Wild-type sequence as string
    substitutions: list
        List of amino acid substitutions of a single variant of the format:
            - Single substitution variant, e.g. variant A123C: ['A123C']
            - Higher variants, e.g. variant A123C/D234E/F345G: ['A123C', 'D234E, 'F345G']
            --> Full substitutions list, e.g.: [['A123C'], ['A123C', 'D234E, 'F345G']]
    fitness_values: list
        List of ints/floats of the variant fitness values, e.g. for two variants: [1.4, 0.8]
    """
    if fitness_values is None:
        fitness_values = np.zeros(len(substitutions)).tolist()
    variants, values, sequences = [], [], []
    for i, var in enumerate(substitutions):  # var are lists of (single or multiple) substitutions
        temp = list(wt_seq)
        name = ''
        separation = 0
        if var == ['WT']:
            name = 'WT'
        else:
            for single_var in var:  # single entries of substitution list
                position_index = int(str(single_var)[1:-1]) - 1 - shift_pos
                new_amino_acid = str(single_var)[-1]
                if str(single_var)[0].isalpha(): # Assertion only possible for format AaPosAa, e.g. A123C
                    assert str(single_var)[0] == temp[position_index], f"Input variant: "\
                        f"{str(single_var)[0]}{position_index + 1}{new_amino_acid}, WT amino "\
                        f"acid variant: {temp[position_index]}{position_index + 1}{new_amino_acid}"
                temp[position_index] = new_amino_acid
                # checking if multiple entries are inside list
                if separation == 0:
                    name += single_var
                else:
                    name += '/' + single_var
                separation += 1
        variants.append(name)
        values.append(fitness_values[i])
        sequences.append(''.join(temp))

    return variants, values, sequences


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
        logger.info(f"Saved encoded CSV file as {os.path.abspath(filename)}...")

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


def check_alignment(wt_seq, pdb_seq, min_block=3):
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    # Higher open/extend gap penalties help prevent "shredding" the alignment
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    
    alignments = aligner.align(wt_seq, pdb_seq)
    if not alignments:
        return None
        
    best_alignment = alignments[0]
    score = best_alignment.score
    identity = score / max(len(wt_seq), len(pdb_seq))

    # Identify the first and last match columns
    target_str = best_alignment[0]
    query_str = best_alignment[1]
    
    first_match_col = None
    last_match_col = None

    # Find the first robust match
    for i in range(len(target_str) - min_block + 1):
        if all(target_str[i+j] == query_str[i+j] and target_str[i+j] != '-' 
               for j in range(min_block)):
            first_match_col = i
            break
            
    # Find last match
    for i in range(len(target_str) - 1, -1, -1):
        if target_str[i] == query_str[i] and target_str[i] != '-':
            last_match_col = i
            break

    if first_match_col is None or last_match_col is None:
        logger.warning("No significant homology between sequences found!")
        return None

    # Map alignment columns to sequence indices
    # coords[0] is WT (target), coords[1] is PDB (query)
    coords = best_alignment.indices
    start_wt = int(coords[0, first_match_col])
    end_wt = int(coords[0, last_match_col]) + 1
    
    start_pdb = int(coords[1, first_match_col])
    end_pdb = int(coords[1, last_match_col]) + 1

    common_seq = wt_seq[start_wt:end_wt]

    mapping = {
        "start_wt": start_wt,
        "end_wt": end_wt,
        "start_pdb": start_pdb,
        "end_pdb": end_pdb,
        "identity": identity,
        "alignment_obj": best_alignment,
        "common_seq": common_seq
    }

    return mapping


def shift_and_trim_vars_seqs(vars_list, seqs_list, alignment_mapping, msa_start):
    """
    Returns four lists: 
    1. pdb_vars: Shifted variant names (PDB-relative)
    2. orig_vars: Original variant names (MSA-relative)
    3. gremlin_seqs: Full-length sequences (len 524) for GREMLIN
    4. trimmed_seqs: Fragment sequences (len 504) for PDB-specific output
    """

    pdb_vars = []
    orig_vars = []
    gremlin_seqs = []
    trimmed_seqs = []

    
    start_offset = alignment_mapping['start_wt']
    end_offset = alignment_mapping['end_wt']

    for var_entry, full_seq in zip(vars_list, seqs_list):
        is_list = isinstance(var_entry, list)
        current_vars = var_entry if is_list else [var_entry]
        
        pdb_sub_vars = []
        is_in_pdb_range = True

        for v in current_vars:
            if v.upper() == "WT":
                pdb_sub_vars.append("WT")
                continue

            orig_pos_abs = int(v[1:-1]) 
            idx_in_msa = orig_pos_abs - msa_start
            
            # Check range
            if start_offset <= idx_in_msa < end_offset:
                # Calculate PDB-relative position
                new_pos_pdb = (orig_pos_abs - msa_start) - start_offset + 1
                
                # Check mutant AA against the sequence
                if full_seq[idx_in_msa] == v[-1]:
                    pdb_sub_vars.append(f"{v[0]}{new_pos_pdb}{v[-1]}")
                else:
                    is_in_pdb_range = False
            else:
                print('orig_pos_abs:', orig_pos_abs)
                print('msa_start:', msa_start)
                print('start_offset:', start_offset)
                new_pos_pdb = (orig_pos_abs - msa_start) - start_offset + 1
                print(f"{v} --> {v[0]}{new_pos_pdb}{v[-1]} not in range!")
                raise RuntimeError
                is_in_pdb_range = False 

        if is_in_pdb_range and pdb_sub_vars:
            # 1. Shifted names
            pdb_vars.append(pdb_sub_vars if is_list else pdb_sub_vars[0])
            # 2. Original names
            orig_vars.append(var_entry)
            # 3. Full sequences (GREMLIN compatible)
            gremlin_seqs.append(full_seq)
            # 4. Trimmed sequences (PDB compatible)
            trimmed_seqs.append(full_seq[start_offset:end_offset])

    return pdb_vars, orig_vars, gremlin_seqs, trimmed_seqs


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


def get_mismatches(seq_a: str, seq_b: str):
    n = 0
    mismatches = ""
    if len(seq_a) != len(seq_b):
        logger.warning("Sequence length's do not match!")
        raise RuntimeError(f"{seq_a}\n{len(seq_a)}\n{seq_b}\n{len(seq_b)}")
    for i_a, aa in enumerate(seq_a):
        if aa != seq_b[i_a]:
            mismatches += f"{aa}{i_a + 1}{seq_b[i_a]},"
            n += 1
    return n, mismatches[:-1]


def extract_positions_from_sequences(wt_seq, variant_seqs):
    """
    Extracts mutated positions by comparing variant sequences to a WT reference.
    Returns an array of tuples representing the mutated indices for grouping.
    """
    groups = []
    
    for seq in variant_seqs:
        # Compare character by character. 
        # Using 0-based indexing here, which is perfectly fine for grouping purposes.
        mutated_positions = tuple(
            i for i, (wt_aa, mut_aa) in enumerate(zip(wt_seq, seq)) 
            if wt_aa != mut_aa
        )
        
        # If the sequence is identical to WT (no mutations), it gets an empty tuple `()`
        groups.append(mutated_positions)
        
    # dtype=object is critical here so numpy doesn't crash if you have 
    # a mix of single-site (length 1 tuple) and multi-site (length >1 tuple) mutants.
    return np.array(groups, dtype=object)


def positional_train_test_split(
        *arrays, 
        wt_sequence, 
        variant_sequences, 
        train_size, 
        random_state, 
        shared_fraction=0.0, 
        verbose=False
):
    """
    Splits data based on mutated positions derived from sequence comparison.
    
    Parameters:
    -----------
    shared_fraction : float, default=0.0
        The fraction (0.0 to 1.0) of unique biological positions that are allowed 
        to bleed/integrate into both training and testing sets.
    """
    # Extract true biological mutation positions
    groups = extract_positions_from_sequences(wt_sequence, variant_sequences)
    raw_groups = groups.copy() # Backup copy for accurate positional statistics

    # Identify all unique biological positions across the dataset
    all_positions = sorted(list(set(p for g in groups for p in g)))
    total_unique_positions = len(all_positions)

    # Randomly sample a fraction of positions to be shared
    shared_set = set()
    if shared_fraction > 0.0 and total_unique_positions > 0:
        # Determine how many positions match the requested fraction
        num_to_share = int(np.round(total_unique_positions * shared_fraction))
        
        # Use a localized RandomState tied to self.seed for reproducible sampling
        rng = np.random.RandomState(random_state)
        shared_sampled = rng.choice(all_positions, size=num_to_share, replace=False)
        shared_set = set(shared_sampled)

    # Inject dummy groups for positions allowed to bypass strict grouping
    if shared_set:
        modified_groups = []
        for idx, g in enumerate(groups):
            # If a variant contains any mutation at a chosen shared position, 
            # bypass strict group containment by giving it a totally unique ID.
            if any(pos in shared_set for pos in g):
                modified_groups.append(f"shared_row_{idx}")
            else:
                modified_groups.append(f"pos_{'_'.join(map(str, g))}" if g else "wt")
        groups = np.array(modified_groups, dtype=str)

    # Calculate train ratio
    if isinstance(train_size, int):
        train_ratio = train_size / len(variant_sequences)
    else:
        train_ratio = float(train_size)

    # Perform the split
    gss = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=random_state)
    all_splits = list(gss.split(variant_sequences, groups=groups))
    train_idx, test_idx = all_splits[0]

    # Enhanced Logging
    if verbose:
        # Extract underlying biological positions actually used in each set
        train_positions = set(p for g in raw_groups[train_idx] for p in g)
        test_positions = set(p for g in raw_groups[test_idx] for p in g)
        
        # Cross-contamination metrics
        actual_overlapping = train_positions.intersection(test_positions)
        
        logger.info(
            f"Positional Split Summary (Shared Fraction: {shared_fraction:.1%}): "
            f" - Target Train Ratio: {train_ratio:.1%}  "
            f" - Total Sequences: {len(variant_sequences)}  "
            f" - Total Bio Positions: {total_unique_positions}  "
            f" - Intentionally Shared: {len(shared_set)} positions  "
            f" - Train Set Size: {len(train_idx)} sequences "
            f"({(len(train_idx)/len(variant_sequences)):.1%})  "
            f" - Test Set Size: {len(test_idx)} sequences "
            f"({(len(test_idx)/len(variant_sequences)):.1%})  "
            f" - Actual Position Overlap between sets: {len(actual_overlapping)} positions"
        )

    # Reconstruct flat array layout
    splits = []
    for arr in arrays:
        if isinstance(arr, np.ndarray):
            splits.extend([arr[train_idx], arr[test_idx]])
        elif hasattr(arr, 'iloc'): 
            splits.extend([arr.iloc[train_idx], arr.iloc[test_idx]])
        else:
            splits.extend([[arr[i] for i in train_idx], [arr[i] for i in test_idx]])

    return splits


def contiguous_train_test_split(*arrays, wt_sequence, variant_sequences, train_size, random_state, verbose=False):
    """
    Splits data by holding out a single continuous block of mutated positions.
    """
    groups = extract_positions_from_sequences(wt_sequence, variant_sequences)
    all_positions = sorted(list(set(p for g in groups for p in g)))
    
    if isinstance(train_size, int):
        train_ratio = train_size / len(variant_sequences)
    else:
        train_ratio = float(train_size)
        
    test_ratio = 1.0 - train_ratio
    
    # 1. Determine how wide the continuous test block should be
    num_test_positions = max(1, int(round(len(all_positions) * test_ratio)))
    
    # 2. Pick a random starting point for the continuous block
    rng = np.random.RandomState(random_state)
    max_start_idx = max(0, len(all_positions) - num_test_positions)
    start_idx = rng.randint(0, max_start_idx + 1)
    
    # 3. Define the contiguous test block
    test_positions = set(all_positions[start_idx : start_idx + num_test_positions])
    
    # 4. Assign sequences
    train_idx, test_idx = [], []
    for i, g in enumerate(groups):
        if any(p in test_positions for p in g):
            test_idx.append(i)
        else:
            train_idx.append(i)
            
    if verbose:
        logger.info(
            f"Contiguous Split Summary: Holding out {num_test_positions} contiguous "
            f"positions starting at idx {start_idx}. Train seqs: {len(train_idx)}, "
            f"Test seqs: {len(test_idx)}"
        )
        
    # 5. Reconstruct arrays
    splits = []
    for arr in arrays:
        if isinstance(arr, np.ndarray):
            splits.extend([arr[train_idx], arr[test_idx]])
        elif hasattr(arr, 'iloc'): 
            splits.extend([arr.iloc[train_idx], arr.iloc[test_idx]])
        else:
            splits.extend([[arr[i] for i in train_idx], [arr[i] for i in test_idx]])
            
    return splits


def modulo_train_test_split(*arrays, wt_sequence, variant_sequences, train_size, random_state, verbose=False):
    """
    Splits data by holding out every N-th mutated position (Modulo split).
    """
    groups = extract_positions_from_sequences(wt_sequence, variant_sequences)
    
    # 1. Get strictly sorted unique biological positions
    all_positions = sorted(list(set(p for g in groups for p in g)))
    
    # 2. Calculate the modulo step based on requested train ratio
    if isinstance(train_size, int):
        train_ratio = train_size / len(variant_sequences)
    else:
        train_ratio = float(train_size)
        
    test_ratio = 1.0 - train_ratio
    
    # E.g., if test_ratio is 0.2 (20%), we want every 5th position (1 / 0.2 = 5)
    modulo_step = max(2, int(round(1.0 / test_ratio)) if test_ratio > 0 else 1)
    
    # 3. Use random_state to pick where the modulo counting starts
    rng = np.random.RandomState(random_state)
    offset = rng.randint(0, modulo_step)
    
    # 4. Define test positions
    test_positions = set(all_positions[offset::modulo_step])
    
    # 5. Assign sequences to train/test based on their mutated positions
    train_idx, test_idx = [], []
    for i, g in enumerate(groups):
        # If any mutation in the sequence hits a modulo test position, hold it out
        if any(p in test_positions for p in g):
            test_idx.append(i)
        else:
            train_idx.append(i)
            
    if verbose:
        logger.info(
            f"Modulo Split Summary: Holding out every {modulo_step}th position "
            f"(offset={offset}). Train seqs: {len(train_idx)}, Test seqs: {len(test_idx)}"
        )
        
    # 6. Reconstruct arrays
    splits = []
    for arr in arrays:
        if isinstance(arr, np.ndarray):
            splits.extend([arr[train_idx], arr[test_idx]])
        elif hasattr(arr, 'iloc'): 
            splits.extend([arr.iloc[train_idx], arr.iloc[test_idx]])
        else:
            splits.extend([[arr[i] for i in train_idx], [arr[i] for i in test_idx]])
            
    return splits


def block_random_train_test_split(*arrays, wt_sequence, variant_sequences, train_size, random_state, block_size=3, verbose=False):
    """
    Splits data by grouping mutated positions into continuous blocks (chunks) 
    and randomly assigning those blocks to train or test sets. 
    Acts as a hybrid of random and contiguous splitting.
    """
    groups = extract_positions_from_sequences(wt_sequence, variant_sequences)
    
    # 1. Get strictly sorted unique biological positions
    all_positions = sorted(list(set(p for g in groups for p in g)))
    
    # 2. Group the positions into continuous chunks
    chunks = [all_positions[i:i + block_size] for i in range(0, len(all_positions), block_size)]
    
    # 3. Calculate target ratios
    if isinstance(train_size, int):
        train_ratio = train_size / len(variant_sequences)
    else:
        train_ratio = float(train_size)
        
    # 4. Determine how many chunks go to the training set
    num_train_chunks = int(round(len(chunks) * train_ratio))
    
    # 5. Randomly shuffle the chunks
    rng = np.random.RandomState(random_state)
    shuffled_chunks = chunks.copy()
    rng.shuffle(shuffled_chunks)
    
    # 6. Assign chunks to sets
    train_chunks = shuffled_chunks[:num_train_chunks]
    test_chunks = shuffled_chunks[num_train_chunks:]
    
    # Flatten the test chunks back into a set of distinct positions
    test_positions = set(p for chunk in test_chunks for p in chunk)
    
    # 7. Assign sequences based on their mutated positions
    train_idx, test_idx = [], []
    for i, g in enumerate(groups):
        # If any mutation hits a blocked test position, hold out the variant
        if any(p in test_positions for p in g):
            test_idx.append(i)
        else:
            train_idx.append(i)
            
    if verbose:
        logger.info(
            f"Block-Random Split Summary (Block Size: {block_size}): "
            f"Total distinct positions: {len(all_positions)} split into {len(chunks)} blocks. "
            f"Train seqs: {len(train_idx)}, Test seqs: {len(test_idx)}"
        )
        
    # 8. Reconstruct flat array layout
    splits = []
    for arr in arrays:
        if isinstance(arr, np.ndarray):
            splits.extend([arr[train_idx], arr[test_idx]])
        elif hasattr(arr, 'iloc'): 
            splits.extend([arr.iloc[train_idx], arr.iloc[test_idx]])
        else:
            splits.extend([[arr[i] for i in train_idx], [arr[i] for i in test_idx]])
            
    return splits


def extract_pdb_coords(pdb_path, target_len=None, chain_id="A", atom_type="CA"):
    """
    Extracts 3D coordinates from a PDB file to create the structural prior array. 
    Aligns directly to the target sequence length.
    
    Args:
        pdb_path (str): Path to the input PDB or CIF structure file.
        target_len (int): The absolute expected sequence length (e.g., 506).
        chain_id (str): The specific target chain to extract. Default is "A".
        atom_type (str): The specific atom to track. "CA" (Alpha Carbon) is the 
                         standard for distance-dependent structural kernels.
                         
    Returns:
        np.ndarray: A coordinate array of shape (target_len, 3)
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein_target", pdb_path)
    
    # Isolate specified model and chain
    model = structure[0]
    if chain_id not in model:
        available_chains = [c.id for c in model.get_chains()]
        raise KeyError(f"Chain '{chain_id}' not found in PDB. Available: {available_chains}")
    
    chain = model[chain_id]
    
    coords = []
    skipped_residues = 0
    
    for residue in chain.get_residues():
        # Filter out heteroatoms (water molecules, ligands, etc.)
        if residue.id[0] != " ":
            continue
            
        if atom_type in residue:
            atom = residue[atom_type]
            # Use .get_coord() to automatically resolve alternative conformations (altloc)
            coords.append(atom.get_coord())
        else:
            # Fallback: If a CA is missing (rare but happens in low-res experimental loops)
            # Fill with a dummy coordinate to preserve absolute index alignment
            coords.append([np.nan, np.nan, np.nan])
            skipped_residues += 1

    coord_array = np.array(coords, dtype=np.float32)
    
    # Handle structural discrepancies vs. sequence models
    if target_len is None:
        logger.warning(
            f"Did not receive PDB `target_len` information and thus can't check for "
            f"matching sequence lengths."
        )
    else:
        if len(coord_array) != target_len:
            logger.warning(
                f"PDB residue count ({len(coord_array)}) mismatches "
                f"target sequence length ({target_len})."
            )

            if len(coord_array) > target_len:
                # Truncate if PDB contains expression tags or trailing unmodeled regions
                coord_array = coord_array[:target_len]
            else:
                # Pad with NaNs if the PDB structure drops unresolved terminal tails
                padding = np.full((target_len - len(coord_array), 3), np.nan)
                coord_array = np.vstack([coord_array, padding])
            
    # Impute any localized missing residues using nearby neighbors so kernel math doesn't fail
    if np.isnan(coord_array).any():
        coord_array = impute_missing_coordinates(coord_array)
    return coord_array


def impute_missing_coordinates(coord_array):
    """Linearly interpolates isolated NaN coordinates to prevent matrix failures."""
    n_positions = coord_array.shape[0]
    collection = []
    for i in range(n_positions):
        if np.isnan(coord_array[i]).any():
            collection.append(i + 1)
            # Look for the closest valid preceding and succeeding coordinates
            prev_idx = next((j for j in range(i - 1, -1, -1) if not np.isnan(coord_array[j]).any()), None)
            next_idx = next((j for j in range(i + 1, n_positions) if not np.isnan(coord_array[j]).any()), None)
            
            if prev_idx is not None and next_idx is not None:
                coord_array[i] = (
                    coord_array[prev_idx] + 
                    (coord_array[next_idx] - coord_array[prev_idx]) * 
                    ((i - prev_idx) / (next_idx - prev_idx))
                )
            elif prev_idx is not None:
                coord_array[i] = coord_array[prev_idx]
            elif next_idx is not None:
                coord_array[i] = coord_array[next_idx]
    if collection:
        logger.info(f"Found NaN in PDB coordinates at residue(s) {collection} (1-indexed)")
    return coord_array

