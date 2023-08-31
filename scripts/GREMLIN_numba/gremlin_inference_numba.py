# GREMLIN in Python using numba
# 16.08.2023, Niklas Siedhoff
# Python-port of GREMLIN_CPP; 
# GREMLIN_CPP is licensed as:

"""
------------------------------------------------------------
"THE BEERWARE LICENSE" (Revision 42):
 <so@g.harvard.edu>  wrote this code. As long as you retain this
 notice, you can do whatever you want with this stuff. If we meet
 someday, and you think this stuff is worth it, you can buy me a
 beer in return. --Sergey Ovchinnikov
 ------------------------------------------------------------
 If you use this code, please cite the following papers:

 Balakrishnan, Sivaraman, Hetunandan Kamisetty, Jaime G. Carbonell,
 Su?In Lee, and Christopher James Langmead.
 "Learning generative models for protein fold families."
 Proteins: Structure, Function, and Bioinformatics 79, no. 4 (2011): 1061-1078.

 Kamisetty, Hetunandan, Sergey Ovchinnikov, and David Baker.
 "Assessing the utility of coevolution-based residue–residue
 contact predictions in a sequence-and structure-rich era."
 Proceedings of the National Academy of Sciences (2013): 201314045.
"""

# Using numba for getting closer to C/C++ performance:
# https://numba.pydata.org/numba-doc/latest/user/performance-tips.html
# https://numba.pydata.org/numba-doc/latest/user/parallel.html#numba-parallel-supported

import numpy as np
from numba import njit, prange
from typing import List


def get_sequences_from_file(file: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    "Get_Sequences" reads (learning and test) .fasta and
    .fasta-like ".fasl" format files and extracts the name,
    the target value and the sequence of the protein.
    Only takes one-liner sequences for correct input.
    See example directory for required fasta file format.
    Make sure every marker (> and ;) is seperated by a
    space ' ' from the value respectively name.
    file: str    : File in FASTA or A2M format
    """
    sequences = []
    values = []
    names_of_mutations = []

    with open(file, 'r') as f:
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

    return np.array(sequences), np.array(names_of_mutations), np.array(values)


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


def a2n_dict(char_alphabet="ARNDCQEGHILKMFPSTWYV-"):
    states = len(char_alphabet)
    a2n = {}
    for a, n in zip(char_alphabet, range(states)):
        a2n[a] = n
    return a2n


def aa2int(aa):
    """convert single aa into numerical integer value, e.g.
    "A" -> 0 or "-" to 21 dependent on char_alphabet"""
    a2n = a2n_dict()
    if aa in a2n:
        return a2n[aa]
    else:  # for unknown characters insert Gap character
        return a2n['-']


def seqs2int(msa_seqs):
    """converts list of sequences to msa"""
    msa_ori = []
    for seq in msa_seqs:
        msa_ori.append([aa2int(aa.upper()) for aa in seq])
    msa_ori = np.array(msa_ori)
    return msa_ori


def filt_gaps(msa_ori, gap_cutoff=0.5):
    """filters alignment to remove gappy positions"""
    tmp = (msa_ori == 21 - 1).astype(float)
    non_gaps = np.where(np.sum(tmp.T, -1).T / msa_ori.shape[0] < gap_cutoff)[0]
    gaps = np.where(np.sum(tmp.T, -1).T / msa_ori.shape[0] >= gap_cutoff)[0]
    return msa_ori[:, non_gaps], gaps


def aa_counts_to_v_ini():
    pass


@njit
def get_1d_position(i_pos: int, aa_pos: int, na: int = 21) -> int:
    """
    positions for 1-bd term
    """
    return i_pos * na + aa_pos


@njit
def get_2d_position(w: int, a: int, b: int, nc: int, na: int = 21) -> int:
    """
    positions for 2-bd term

    w: int from range 0, 1, ..., pair_size-1
    a: int from range 0, 1, ..., na-1 (=20)
    b: int from range 0, 1, ..., na-1 (=20)
    nc: Number of trimmed MSA columns for shifting position by N1 = nc * na
    """
    return nc * na + w * na * na + a * na + b


@njit
def get_h_entropy(msa_trimmed, weights, na : int = 21):
    """
    Transform AA counts to entropy = counts * eff[sequence_n]
    """
    nr = msa_trimmed.shape[0]
    nc = msa_trimmed.shape[1]
    freq = np.zeros(nc * na)
    for i in range(nc):
        for n in range(nr):
            d = get_1d_position(i, msa_trimmed[n][i])
            freq[d] += weights[n]
    return freq


@njit
def get_all_pairs(nc):
    """
    Coupling pairs of two, resulting in shape (int(nc * (nc - 1) / 2), 2)
    """
    pairs = []
    for i in range(nc):
        for j in range(i + 1, nc):
            pairs.append([i, j])
    return pairs


@njit
def get_eff_weights(msa_trimmed: np.ndarray, eff_cutoff: float = 0.8):
    """
    Weights alignment sequences according to sequence similarity:
    If sequences are too identical, they get down-weighted.

    :return:
    """
    nr = msa_trimmed.shape[0]
    nc = msa_trimmed.shape[1]
    chk = nc * eff_cutoff
    N_ = np.ones(nr)
    eff_weights = np.ones(nr)
    for n in prange(nr):
        w = N_[n]
        for m in range(n+1, nr):
            hm = 0
            for i in range(nc):
                if msa_trimmed[n][i] == msa_trimmed[m][i]:
                    hm += 1
            if hm > chk:
                N_[m] += 1
                w += 1
        eff_weights[n] = 1.0 / w
    neff = np.sum(eff_weights)
    return neff, eff_weights


@njit
def eval_v(
        x: np.ndarray,
        msa_trimmed: np.ndarray,
        na: int = 21,
        lam_v=0.01
) -> tuple[float, np.ndarray]:
    """
    1-bd term
    :return: float
    """
    # nr = msa_trimmed.shape[0]
    nc = msa_trimmed.shape[1]
    g = np.zeros(nc * na)  # Reset g
    neff, weights = get_eff_weights(msa_trimmed)
    f = get_h_entropy(msa_trimmed, weights)

    fx = 0.0
    reg = 0.0
    PC = np.zeros(shape=(nc, na))  # (P)robability of (C)olumn entries

    for i in range(nc):
        d = get_1d_position(i, 0)
        for a in range(na):
            PC[i][a] += x[d]
            d += 1

    for i in range(nc):
        # compute Z for each column
        Z = 0.0
        for a in range(na):
            Z += np.exp(PC[i][a])
        Z = np.log(Z)  # The natural logarithm is logarithm in base e
        for a in range(na):
            fx += f[get_1d_position(i, a)] * (PC[i][a] - Z)
            PC[i][a] = np.exp(PC[i][a] - Z)
    for i in range(nc):
        d = get_1d_position(i, 0)
        for a in range(na):
            g[d] += (PC[i][a] * neff) - f[get_1d_position(i, a)] + (lam_v * 2 * x[d])  # Changing of gradient
            reg += lam_v * np.power(x[d], 2)
            d += 1

    return -1.0 * (fx - reg), g


@njit(parallel=True)
def eval_vw(
        x: np.ndarray,
        msa_trimmed: np.ndarray,
        na: int = 21,
        lam_v: float = 0.01
) -> tuple[float, np.ndarray]:
    """
    Updates
        (g)radient (array)
        and fx (float)
    according to given input
        x (array).
    N1 = nc * na
    N2 = N1 + pair_size * na * na
       = N1 +
    """
    nr = msa_trimmed.shape[0]
    nc = msa_trimmed.shape[1]
    pairs = get_all_pairs(nc)
    pair_size = int(nc * (nc - 1) / 2)
    N1 = nc * na
    N2 = N1 + pair_size * na * na
    g = np.zeros(N2)  # Reset g
    neff, weights = get_eff_weights(msa_trimmed)
    lam_w = lam_v * (nc - 1.0) * (na - 1.0)

    fx = 0.0
    reg = 0.0
    PCN = np.zeros(shape=(nr, nc, na))  # (P)robability of (C)olumn entries?!

    for n in prange(nr):
        # precompute sum(V+W) for each position "i" and amino acids "a"
        # assuming all other positions are fixed
        PC = np.zeros(shape=(nc, na))
        # for each position i
        for i in range(nc):
            # for each amino acid
            for a in range(na):
                # 1-body-term
                PC[i][a] += x[get_1d_position(i, a)]
        if True:  # if (N == msa.N2)
            for w in range(pair_size):
                i = pairs[w][0]
                j = pairs[w][1]
                xni = msa_trimmed[n][i]
                xnj = msa_trimmed[n][j]
                for a in range(na):
                    PC[i][a] += x[get_2d_position(w, a, xnj, nc)]
                    PC[j][a] += x[get_2d_position(w, xni, a, nc)]
        for i in range(nc):
            # compute local Z
            Z = 0.0
            for a in range(na):
                Z += np.exp(PC[i][a])
            Z = np.log(Z)

            # compute fx
            xni = msa_trimmed[n][i]
            fx += (PC[i][xni] - Z) * weights[n]

            # needed for (g)radient calculation
            for a in range(na):
                PCN[n][i][a] = np.exp(PC[i][a] - Z) * weights[n]
    # compute (g)radient for 1bd
    if True:  # if (N >= msa.N1)
        for i in prange(nc):
            for n in range(nr):
                xni = msa_trimmed[n][i]
                g[get_1d_position(i, xni)] -= weights[n]
                for a in range(na):
                    g[get_1d_position(i, a)] += PCN[n][i][a]
    #  compute (g)radient for 2bd
    if True:  # if (N == msa.N2)
        for w in prange(pair_size):
            i = pairs[w][0]
            j = pairs[w][1]
            for n in range(nr):
                xni = msa_trimmed[n][i]
                xnj = msa_trimmed[n][j]
                g[get_2d_position(w, xni, xnj, nc)] -= 2.0 * weights[n]
                for a in range(na):
                    g[get_2d_position(w, a, xnj, nc)] += PCN[n][i][a]
                    g[get_2d_position(w, xni, a, nc)] += PCN[n][j][a]
    # compute (reg)ularization and (g)raident for 1bd
    if True:  # if (N >= msa.N1)
        for d in prange(N1):
            reg += lam_v * np.power(x[d], 2)
            g[d] += lam_v * 2.0 * x[d]
    # compute (reg)ularization and (g)radient for 2bd
    if True:  # if (N == msa.N2)
        for d in prange(N1, N2):
            reg += lam_w * np.power(x[d], 2)
            g[d] += lam_w * 2.0 * x[d]
    # flip direction, since we are passing function to a minimizer
    return -1.0 * (fx - reg), g


@njit
def deep_copy(arr: list | np.ndarray):
    """
    Just makes a copy similar to the copy (deepcopy) module.
    Required for making a copy of 1D-arrays using numba.
    """
    ret: List[float] = list()
    for value in arr:
        ret.append(value)
    return np.array(ret)


@njit(parallel=True)
def lbfgs(msa_trimmed, func, mode: str = 'vw', max_iter: int = 100, na: int = 21):
    """
    Limited-memory Broyden–Fletcher–Goldfarb–Shanno (LBFGS)
    ---------------------------------------------------------------------------
    Adopted from: GREMLIN_CPP, https://github.com/sokrypton/GREMLIN_CPP, that is
    Adopted from: https://github.com/js850/lbfgs_cpp
    "THE BEER-WARE LICENSE" (Revision 42):
     <js850@camsa.ac.uk> wrote this function. As long as you retain this notice
     you can do whatever you want with this stuff. If we meet some day, and you
     think this stuff is worth it, you can buy me a beer in return Jacob Stevenson
     ---------------------------------------------------------------------------
    modified to remove convergence criteria, will continue until max_iter
    modified to remove maxstep check
     ---------------------------------------------------------------------------

    func: Target function returning a scalar value to be minimized
    :return:
    """
    # nr = msa_trimmed.shape[0]
    nc = msa_trimmed.shape[1]

    max_f_rise = 1e-4
    H0 = 0.1
    # maxstep = 0.2
    # pairs = get_all_pairs(nc)
    pair_size = int(nc * (nc - 1) / 2)
    N1 = nc * na
    N2 = N1 + pair_size * na * na

    if mode == 'v':
        N = nc * na
    elif mode == 'vw':
        N = N2
    else:
        raise SystemError("Unknown mode.")

    M = 5
    y = np.zeros(shape=(M, N))
    s = np.zeros(shape=(M, N))
    x_new = np.zeros(N)
    rho = np.zeros(M)
    step = np.zeros(N)

    # Compute target (f)unction value
    fx_new, g_new = func(x_new, msa_trimmed)

    print('lbfgs::iter S_S fx: ', fx_new, 'gnorm:', np.linalg.norm(g_new))

    for iteration in range(max_iter):
        g_old = deep_copy(g_new)
        x_old = deep_copy(x_new)
        fx_old = fx_new
        if iteration == 0:
            gnorm = np.linalg.norm(g_old)
            if gnorm > 1.0:
                gnorm = 1.0 / gnorm
            for n in range(N):
                step[n] = -gnorm * H0 * g_old[n]
        else:  # iteration > 0
            step = deep_copy(g_old)
            jmin = iteration - M
            if jmin < 0:
                jmin = 0
            jmax = iteration

            # i = 0
            # beta = 0.0
            alpha = np.zeros(M)

            for j in range(jmax - 1, jmin - 1, -1):
                i = j % M
                alpha[i] = rho[i] * np.dot(s[i], step)
                for n in range(N):
                    step[n] -= alpha[i] * y[i][n]
            # Scale the step size by H0
            for n in range(N):
                step[n] *= H0
            for j in range(jmin, jmax):
                i = j % M
                beta = rho[i] * np.dot(y[i], step)
                for n in range(N):
                    step[n] += s[i][n] * (alpha[i] - beta)
            # invert the step to point downhill
            for n in range(N):
                step[n] *= -1
        ###############################################
        # backtracking_linesearch
        ###############################################
        # fnew = 0.0
        # if the step is pointing uphill, invert it
        if np.dot(step, g_old) > 0.0:
            for n in range(N):
                step[n] *= -1
        attempt = 0
        factor = 1.0
        # stepsize = np.linalg.norm(step)

        step_sum = 0.0
        for step_i in step:
            step_sum += step_i
        # make sure the step is no larger than maxstep
        # if (factor * stepsize > maxstep){factor = maxstep/stepsize;}
        for nred in range(10):
            attempt += 1
            for n in range(N):
                x_new[n] = x_old[n] + factor * step[n]
            fx_new, g_new = func(x_new, msa_trimmed)  # new func calculation
            df = fx_new - fx_old
            if df < max_f_rise:
                break
            else:
                factor /= 10.0
        # stepsize = stepsize * factor
        ###############################################
        # update_memory
        ###############################################
        klocal = iteration % M
        for n in range(N):
            y[klocal][n] = g_new[n] - g_old[n]
            s[klocal][n] = x_new[n] - x_old[n]
        ys = np.dot(y[klocal], s[klocal])
        if ys == 0.0:
            ys = 1.0
        rho[klocal] = 1.0 / ys

        yy = np.dot(y[klocal], y[klocal])
        if yy == 0.0:
            yy = 1.0
        H0 = ys / yy
        try:
            if (iteration + 1) % int(max_iter / 10) == 0:
                print(iteration + 1, '/', max_iter, ' : ', fx_new)
        except:  # bare 'except' used for numba functionality
            print(iteration + 1, '/', max_iter, ' : ', fx_new)

    return x_new


@njit
def cg(msa_trimmed, func, mode: str = 'vw', max_iter: int = 100, na: int = 21):
    """
    Nonlinear Conjugate Gradient (CG)
    ---------------------------------------------------------------------------
    Adopted from: GREMLIN_CPP, https://github.com/sokrypton/GREMLIN_CPP, that is
    Adopted from: https://bitbucket.org/soedinglab/libconjugrad
    CCMpred is released under the GNU Affero General Public License v3 or later.
    ---------------------------------------------------------------------------
    modified to remove convergence criteria, will continue until max_iter
    ---------------------------------------------------------------------------

    :return:
    """
    # nr = msa_trimmed.shape[0]
    nc = msa_trimmed.shape[1]
    pair_size = int(nc * (nc - 1) / 2)
    N1 = nc * na
    N2 = N1 + pair_size * na * na

    if mode == 'v':
        N = N1
    elif mode == 'vw':
        N = N2
    else:
        raise SystemError("Unknown mode.")

    # epsilon = 1e-5
    ftol = 1e-4
    wolfe = 0.1
    alpha_mul = 0.5
    max_line = 10

    s = np.zeros(N)

    gnorm_old = 0.0
    alpha_old = 0.0
    dg_old = 0.0

    x = np.zeros(N)

    fx, g = func(x, msa_trimmed)

    gnorm = np.dot(g, g)

    # dg = 0.0
    alpha = 1.0 / np.sqrt(gnorm)

    print("# cg::iter S_S fx: ", fx, " gnorm: ", np.sqrt(gnorm))

    for iteration in range(max_iter):
        if iteration == 0:
            for n in range(N):
                s[n] = -g[n]
            dg = np.dot(s, g)
        else:  # iteration > 0
            # Fletcher-Reeves
            beta = gnorm / gnorm_old
            for n in range(N):
                s[n] = s[n] * beta - g[n]
            dg = np.dot(s, g)
            alpha = alpha_old * dg_old / dg
        #########################################
        # Linesearch
        #########################################
        attempts = 0
        dg_ini = dg
        dg_test = dg_ini * ftol
        fx_ini = fx
        old_alpha = 0.0
        for line in range(max_line):
            attempts += 1
            step = alpha - old_alpha
            for n in range(N):
                x[n] += s[n] * step
            fx_step, g = func(x, msa_trimmed)
            if fx_step <= fx_ini + alpha * dg_test:
                if np.dot(s, g) < wolfe * dg_ini:
                    fx = fx_step
                    break
            old_alpha = alpha
            alpha *= alpha_mul
        #########################################
        gnorm_old = gnorm
        gnorm = np.dot(g, g)
        alpha_old = alpha
        dg_old = dg
        try:
            if (iteration + 1) % int(max_iter / 10) == 0:
                print(iteration + 1, '/', max_iter, ' : ', fx)
        except:  # bare 'except' used for numba functionality
            print(iteration + 1, '/', max_iter, ' : ', fx)
    return x


def get_seqs_onehot_1bd(seqs_int):
    new_seqs_oh = []
    for arr in seqs_int:
        new_seq_oh = np.zeros(len(arr) * 21, dtype=int)
        for i, a, in enumerate(arr):
            d = get_1d_position(i, a)
            new_seq_oh[d] += 1
        new_seqs_oh.append(new_seq_oh)
    return np.array(new_seqs_oh)


def get_pair_size(seqs_int):
    for arr in seqs_int[:1]:
        c = 0
        for a_i in range(len(arr)):
            for b_i in range(a_i + 1, len(arr)):
                c += 1
    return c


@njit
def oh_1bd_predict(seqs_int, x_opt, na: int = 21):
    nc = seqs_int.shape[1]
    N1 = nc * na
    y_pred_all: List[float] = list()
    for arr in seqs_int:
        new_seq_oh = np.zeros(N1)
        for i, a, in enumerate(arr):
            d = get_1d_position(i, a)
            new_seq_oh[d] += 1
        # Predict here directly to save memory instead
        # of storing whole one-hot vector
        y_pred = np.sum(new_seq_oh * x_opt)
        y_pred_all.append(y_pred)
    return y_pred_all


@njit(parallel=True)
def oh_2bd_predict(seqs_int, x_opt, na: int = 21):
    nc = seqs_int.shape[1]
    pair_size = int(nc * (nc - 1) / 2)
    N2 = nc * na + pair_size * na * na
    y_pred_all: List[float] = list()
    pairs = get_all_pairs(nc)
    for arr in seqs_int:
        new_seq_oh = np.zeros(N2)
        for i, a, in enumerate(arr):
            d = get_1d_position(i, a)
            new_seq_oh[d] += 1
        for w in prange(pair_size):
            i = pairs[w][0]
            j = pairs[w][1]
            xni = arr[i]
            xnj = arr[j]
            new_seq_oh[get_2d_position(w, xni, xnj, nc)] += 1
        # Predict here directly to save memory instead
        # of storing whole one-hot vector
        y_pred = np.sum(new_seq_oh * x_opt)
        y_pred_all.append(y_pred)
    return y_pred_all


@njit
def oh_1bd_1d_encode(seqs_int, x_opt):
    """
    Encoding the input sequences (amino acids to integer-transformed
    trimmed MSA sequences).
    As V, i.e., 1-body-transformed, sequences require much memory to
    be stored, encoding leads to len(sequence) encoded sequences, i.e.,
    adding up sequence position-dependent probabilities.

    :param seqs_int:
    :return:
    """
    x_oh_pos_all: List[float] = list()
    for arr in seqs_int:
        x_oh_pos = np.zeros(np.shape(seqs_int)[1])
        for i, a, in enumerate(arr):
            d = get_1d_position(i, a)
            x_oh_pos[i] += x_opt[d]
        x_oh_pos_all.append(x_oh_pos)

    return x_oh_pos_all



@njit(parallel=True)
def oh_2bd_1d_encode(seqs_int, x_opt):
    """
    Encoding the input sequences (amino acids to integer-transformed
    trimmed MSA sequences).
    As V+W, i.e., 2-body-transformed, sequences require much memory to
    be stored, encoding leads to len(sequence) encoded sequences, i.e.,
    adding up sequence position-dependent probabilities.

    :param seqs_int:
    :return:
    """
    nc = seqs_int.shape[1]
    pairs = get_all_pairs(nc)
    pair_size = int(nc * (nc - 1) / 2)

    x_oh_pos_all: List[float] = list()
    for arr in seqs_int:
        x_oh_pos = np.zeros(np.shape(seqs_int)[1])
        for i, a, in enumerate(arr):
            d = get_1d_position(i, a)
            x_oh_pos[i] += x_opt[d]
        for w in prange(pair_size):
            i = pairs[w][0]
            j = pairs[w][1]
            xni = arr[i]
            xnj = arr[j]
            d = get_2d_position(w, xni, xnj, nc)
            x_oh_pos[i] += x_opt[d]
        x_oh_pos_all.append(x_oh_pos)

    return x_oh_pos_all


def aa_count_predict(msa_trimmed, sequences_int_trimmed):
    """
    Simple encoding based on amino acid counts (MSA column-based counts/frequencies).

    :param msa_trimmed: Multiple sequence alignment (MSA) in trimmed form.
    :param sequences_int_trimmed: Int-converted input sequences to be used
        for fitness prediction.

    :return: Predicted fitness values of input sequences.
    """
    onehot_cat_msa = np.eye(21)[msa_trimmed]
    aa_counts = np.sum(onehot_cat_msa, axis=0)
    sequences_int_trimmed_onehot = np.eye(21)[sequences_int_trimmed]
    tmp = []
    for seq_oh in sequences_int_trimmed_onehot:
        tmp.append(seq_oh.flatten())
    sequences_int_trimmed_onehot = np.array(tmp)
    x_aac_pred = sequences_int_trimmed_onehot * aa_counts.flatten()
    y_pred_aac = np.sum(x_aac_pred, axis=1)
    return y_pred_aac


# def sum_encoded_seqs(x_seqs):
#     return np.sum(x_seqs, axis=1)
#
#
# def oh_2bd_pred(seqs_int, x_opt):
#     nc = seqs_int.shape[1]
#     pairs = get_all_pairs(nc)
#     pair_size = int(nc * (nc - 1) / 2)
#     x_oh_pos_all: List[float] = list(list())
#     for arr in seqs_int:
#         sum_x = 0.0
#         for i, a, in enumerate(arr):
#             d = get_1d_position(i, a)
#             sum_x += x_opt[d]
#         for w in prange(pair_size):
#             i = pairs[w][0]
#             j = pairs[w][1]
#             xni = arr[i]
#             xnj = arr[j]
#             d = get_2d_position(w, xni, xnj, nc)
#             sum_x += x_opt[d]
#         x_oh_pos_all.append(sum_x)
#     return x_oh_pos_all
