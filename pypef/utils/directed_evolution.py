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
Modules for performing random evolution walks
similar as presented by Biswas et al.
"""

import os
import re
import random

import matplotlib.pyplot as plt
import numpy as np
import warnings
from adjustText import adjust_text

from pypef.ml.regression import predict
from pypef.dca.hybrid_model import predict_directed_evolution

# ignoring warnings of scikit-learn regression
warnings.filterwarnings(action='ignore', category=RuntimeWarning, module='sklearn')
warnings.filterwarnings(action='ignore', category=UserWarning, module='sklearn')


class DirectedEvolution:
    # Class attributes (None)
    def __init__(  # Instance attributes
            self,
            ml_or_hybrid: str,
            encoding: str,
            s_wt: str,
            y_wt: float,
            single_vars: list,
            num_iterations: int,
            num_trajectories: int,
            amino_acids: list,
            temp: float,
            path: str,
            model: str = None,
            no_fft=False,
            dca_encoder=None,
            usecsv=False,
            csvaa=False,
            negative=False
    ):
        """
        Runs in silico directed evolution and plots and writes trajectories.

        Parameters
        ----------
        ml_or_hybrid: str
            'ml' or 'hybrid'
        encoding: str
            'aaidx' or 'dca'
        s_wt: str
            WT sequence, s_wt = get_wt_sequence(arguments['--wt'])
        y_wt: float
            WT fitness, y_wt = arguments['--y_wt']
        single_vars:  list
            single substituted protein variants; used for recombination
            of variants. Obtained from the CSV file with get_variants:
            single_variants, single_values, higher_variants, higher_values = \
                get_variants(df, amino_acids, s_wt)
        num_iterations: int
            Number of tried steps in the evolution process
        num_trajectories: int
            Number of independent evolutionary trajectories
        amino_acids: list
            Usually the 20 standard amino acids
        temp: float
            (Boltzmann) 'Temperature' of the Metropolis-Hastings algorithm for
            accepting new trajectory variants
        path: str
            Just current working directory (os.getcwd())
        model: str
            Loaded Pickle file for regression/hybrid modeling.
        no_fft: bool
            If True, not using FFT for AAindex-based encoding
        dca_encoder = None or DCAEncoding object
            dca_encoder = DCAEncoding(
                  params_file=arguments['--plmc_params'],
                  separator=arguments['--sep']
            )
        usecsv: bool
            Using only CSV variants for recombination (but all 20 amino acids)
        csvaa: bool
            Using only CSV variants for recombination (but all amino acids that
            are present CSV)
        negative: bool
            More negative variants define improved variants
        """
        self.ml_or_hybrid = ml_or_hybrid
        self.encoding = encoding
        self.s_wt = s_wt
        self.y_wt = y_wt
        self.single_vars = single_vars
        self.num_iterations = num_iterations
        self.num_trajectories = num_trajectories
        self.amino_acids = amino_acids
        self.temp = temp
        self.path = path
        self.model = model
        self.no_fft = no_fft  # for AAidx only
        self.dca_encoder = dca_encoder
        self.usecsv = usecsv
        self.csvaa = csvaa
        self.negative = negative

        self.de_step_counter = 0  # DE steps
        self.traj_counter = 0  # Trajectory counter

    def mutate_sequence(
            self,
            seq: str,
            prev_mut_loc: int
    ):
        """
        Parameters
        ----------
        seq: str,
            Initial sequence to be mutated, must not be WT Seq but can
            also itself be already substituted (iterative sequence substitutions)
        prev_mut_loc: int
            Previous position mutated, new position will be randomly chosen within
            a range, by default: new_pos = previous_pos +- 8

        Produces a mutant sequence (integer representation), given an initial sequence
        and the previous position of mutation.

        """
        try:
            os.mkdir('EvoTraj')
        except FileExistsError:
            pass

        var_seq_list = []
        rand_loc = random.randint(prev_mut_loc - 8, prev_mut_loc + 8)  # find random position to mutate
        while (rand_loc <= 0) or (rand_loc >= len(seq)):
            rand_loc = random.randint(prev_mut_loc - 8, prev_mut_loc + 8)
        aa_list = self.amino_acids
        if self.usecsv:     # Only perform directed evolution on positional csv variant data,
            pos_list = []   # else: aa_list = amino_acids
            aa_list = []    # overwrite aa_list = self.amino_acids
            for aa_positions_aa in self.single_vars:  # getting each single variant, e.g. of [['L215F'], ['A217N']]
                for variant in aa_positions_aa:  # just unpacking the variant, e.g. ['L215F'] -> 'L215F'
                    pos_int = int(re.findall(r"\d+", variant)[0])
                    if pos_int not in pos_list:
                        pos_list.append(pos_int)
                    if self.csvaa:
                        new_aa = str(variant[-1:])  # new AA from known variant, e.g. 'F' from 'L215F'
                        if new_aa not in aa_list:
                            aa_list.append(new_aa)
                    else:
                        aa_list = self.amino_acids  # new AA can be any of the 20 standard AA's
            # Select closest position to single AA positions:
            # However, this means that it is more probable that starting with lower substitution
            # positions new substitution positions will likely be shifted towards higher positions.
            # And for higher substitution positions new substitutions will likely be at lower positions.
            absolute_difference_function = lambda list_value: abs(list_value - rand_loc)
            try:
                closest_loc = min(pos_list, key=absolute_difference_function)
            except ValueError:
                raise ValueError("No positions for recombination found. Likely no single "
                                 "substituted variants were found in provided .csv file.")
            rand_loc = closest_loc - 1   # - 1 as position is shifted by one when starting with 0 index
        rand_aa = random.choice(aa_list)  # find random amino acid to mutate to
        seq_list = list(seq)
        seq_list[rand_loc] = rand_aa  # update sequence to have new amino acid at randomly chosen position
        seq_m = ''.join(seq_list)
        var = str(rand_loc + 1) + str(rand_aa)
        var_seq_list.append((var, seq_m))  # list of tuples

        return var_seq_list

    @staticmethod
    def assert_trajectory_sequences(v_traj, s_traj):
        """
        Making sure that sequence mutations have been introduced correctly
        (for last sequence only).
        """
        for i, variant in enumerate(v_traj[1:]):  # [1:] as not checking for WT
            variant_position = int(re.findall(r"\d+", variant)[0]) - 1
            variant_amino_acid = str(variant[-1])
            assert variant_amino_acid == s_traj[i+1][variant_position]  # checking AA of last trajactory sequence

    def in_silico_de(self):
        """
        Perform directed evolution by randomly selecting a sequence
        position for substitution and randomly choose the amino acid
        to substitute to. New sequence gets accepted if meeting the
        Metropolis criterion and will be taken for new substitution
        iteration. Metropolis-Hastings-driven directed evolution,
        similar to Biswas et al.:
        Low-N protein engineering with data-efficient deep learning,
        see https://github.com/ivanjayapurna/low-n-protein-engineering/tree/master/directed-evo
        """
        # iterate through the trial mutation steps for the directed evolution trajectory
        # m = 1 (only 1 mutation per step) instead of (np.random.poisson(2) + 1)
        v_traj, s_traj, y_traj = [], [], []
        v_traj.append('WT')
        y_traj.append(self.y_wt)
        s_traj.append(self.s_wt)
        accepted = 0
        for iteration in range(self.num_iterations):  # num_iterations
            self.de_step_counter = iteration

            if accepted == 0:
                prior_mutation_location = random.randint(0, len(self.s_wt))  # not really "prior" as first
            else:  # get prior mutation position
                prior_mutation_location = int(re.findall(r"\d+", v_traj[-1])[0])
            prior_y = y_traj[-1]  # prior y, always at [-1]
            prior_sequence = s_traj[-1]  # prior sequence, always at [-1]

            new_var_seq = self.mutate_sequence(
                seq=prior_sequence,
                prev_mut_loc=prior_mutation_location
            )

            new_variant = new_var_seq[0][0]  # int + string char, e.g. '17A'
            new_full_variant = str(self.s_wt[int(new_variant[:-1])-1]) + new_variant  # full variant name, e.g. 'F17A'
            new_sequence = new_var_seq[0][1]
            # encode and predict new sequence fitness
            if self.ml_or_hybrid == 'ml':
                predictions = predict(  # AAidx, OneHot, or DCA-based pure ML prediction
                    path=self.path,
                    model=self.model,
                    encoding=self.encoding,
                    variants=np.atleast_1d(new_full_variant),
                    sequences=np.atleast_1d(new_sequence),
                    no_fft=self.no_fft,
                    dca_encoder=self.dca_encoder
                )

            else:  # hybrid modeling and prediction
                predictions = predict_directed_evolution(
                    encoder=self.dca_encoder,
                    variant=self.s_wt[int(new_variant[:-1]) - 1] + new_variant,
                    hybrid_model_data_pkl=self.model
                )

            if predictions == 'skip':  # skip if variant cannot be encoded by DCA-based encoding technique
                continue
            new_y, new_var = predictions[0][0], predictions[0][1]  # new_var == new_variant nonetheless
            # probability function for trial sequence
            # The lower the fitness (y) of the new variant, the higher are the chances to get excluded
            with warnings.catch_warnings():  # catching Overflow warning
                warnings.simplefilter("ignore")
                try:
                    boltz = np.exp(((new_y - prior_y) / self.temp), dtype=np.longfloat)
                    if self.negative:
                        boltz = np.exp((-(new_y - prior_y) / self.temp), dtype=np.longfloat)
                except OverflowError:
                    boltz = 1
            p = min(1, boltz)
            rand_var = random.random()  # random float between 0 and 1
            if rand_var < p:  # Metropolis-Hastings update selection criterion, else do nothing (do not accept variant)
                v_traj.append(new_var)       # update the variant naming trajectory
                y_traj.append(new_y)         # update the fitness trajectory records
                s_traj.append(new_sequence)  # update the sequence trajectory records
                accepted += 1

        self.assert_trajectory_sequences(v_traj, s_traj)

        return v_traj, s_traj, y_traj

    def run_de_trajectories(self):
        """
        Runs the directed evolution by addressing the in_silico_de
        function and plots the evolution trajectories.
        """
        v_records = []  # initialize list of sequence variant names
        s_records = []  # initialize list of sequence records
        y_records = []  # initialize list of fitness score records
        #   i = counter, iterate through however many mutation trajectories we want to sample
        for i in range(self.num_trajectories):
            self.traj_counter = i
            # call the directed evolution function, outputting the trajectory
            # sequence and fitness score records
            v_traj, s_traj, y_traj = self.in_silico_de()
            v_records.append(v_traj)  # update variant naming full mutagenesis trajectory
            s_records.append(s_traj)  # update the sequence full mutagenesis trajectory
            y_records.append(y_traj)  # update the fitness full mutagenesis trajectory

        return s_records, v_records, y_records

    def plot_trajectories(self):
        """
        Plots evolutionary trajectories and saves steps
        in CSV file.
        """
        s_records, v_records, y_records = self.run_de_trajectories()
        # Idea: Standardizing DCA-HybridModel predictions as just trained by Spearman's rho
        # e.g., meaning that fitness values could differ only at the 6th decimal place and only
        # predicted fitness ranks matter and not associated fitness values
        fig, ax = plt.subplots()  # figsize=(10, 6)
        ax.locator_params(integer=True)
        y_records_ = []
        for i, fitness_array in enumerate(y_records):
            ax.plot(np.arange(1, len(fitness_array) + 1, 1), fitness_array,
                    '-o', alpha=0.7, markeredgecolor='black', label='EvoTraj' + str(i + 1))
            y_records_.append(fitness_array)
        label_x_y_name = []
        traj_max_len = 0
        for i, v_record in enumerate(v_records):  # i = 1, 2, 3, .., ; v_record = variant label array
            for j, v in enumerate(v_record):      # j = 1, 2, 3, ..., ; v = variant name; y_records[i][j] = fitness
                if len(v_record) > traj_max_len:
                    traj_max_len = len(v_record)
                if i == 0:                      # j + 1 -> x-axis position shifted by 1
                    label_x_y_name.append(ax.text(j + 1, y_records_[i][j], v, size=9))
                else:
                    if v != 'WT':  # only plot 'WT' name once at i == 0
                        label_x_y_name.append(ax.text(j + 1, y_records_[i][j], v, size=9))
        adjust_text(label_x_y_name, only_move={'points': 'y', 'text': 'y'}, force_points=0.5)
        ax.legend()
        plt.xticks(np.arange(1,  traj_max_len + 1, 1), np.arange(1, traj_max_len + 1, 1))

        plt.ylabel('Predicted fitness')
        plt.xlabel('Mutation trial steps')
        plt.savefig(str(self.model) + '_DE_trajectories.png', dpi=500)
        plt.clf()

        with open(os.path.join('EvoTraj', 'Trajectories.csv'), 'w') as file:
            file.write('Trajectory;Variant;Sequence;Fitness\n')
            for i in range(self.num_trajectories):
                v_records_str = str(v_records[i])[1:-1].replace("'", "")
                s_records_str = str(s_records[i])[1:-1].replace("'", "")
                y_records_str = str(y_records[i])[1:-1]
                file.write(f'{i+1};{v_records_str};{s_records_str};{y_records_str}\n')
