# PyPEF - Pythonic Protein Engineering Framework
# https://github.com/niklases/PyPEF

"""
Modules for performing random evolution walks
similar as presented by Biswas et al.
"""


from __future__ import annotations
import os
import re
import random

import matplotlib.pyplot as plt
import numpy as np
import warnings
from adjustText import adjust_text
import logging
logger = logging.getLogger('pypef.utils.directed_evolution')

from pypef.ml.regression import predict
from pypef.hybrid.hybrid_model import predict_directed_evolution

# ignoring warnings of scikit-learn regression
warnings.filterwarnings(action='ignore', category=RuntimeWarning, module='sklearn')
warnings.filterwarnings(action='ignore', category=UserWarning, module='sklearn')


class DirectedEvolution:
    # Class attributes (None)
    def __init__(
            self,
            ml_or_hybrid: str,
            encoding: str,
            s_wt: str,
            single_vars: list,
            num_iterations: int,
            num_trajectories: int,
            amino_acids: list,
            temp: float,
            path: str,
            model: str = None,
            no_fft: bool = False,
            dca_encoder: str | None = None,
            usecsv: bool = False,
            csvaa: bool = False,
            negative: bool = False
    ):
        """
        Runs in silico directed evolution and plots and writes trajectories.
        """
        self.ml_or_hybrid = ml_or_hybrid
        self.encoding = encoding
        self.s_wt = s_wt
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
        logger.info(f"Directed evolution acceptance \"temperature\": {self.temp}")

    def mutate_sequence(
            self,
            seq: str,
            prev_mut_loc: int
    ):
        try:
            os.mkdir('EvoTraj')
        except FileExistsError:
            pass

        var_seq_list = []
        rand_loc = random.randint(prev_mut_loc - 8, prev_mut_loc + 8)  # find random position to mutate
        while (rand_loc <= 0) or (rand_loc >= len(seq)):
            rand_loc = random.randint(prev_mut_loc - 8, prev_mut_loc + 8)
        aa_list = self.amino_acids
        if self.usecsv:     # Only perform directed evolution on positional csv variant data
            pos_list = []   
            aa_list = []    
            for aa_positions_aa in self.single_vars:  # getting each single variant
                for variant in aa_positions_aa:
                    pos_int = int(re.findall(r"\d+", variant)[0])
                    if pos_int not in pos_list:
                        pos_list.append(pos_int)
                    if self.csvaa:
                        new_aa = str(variant[-1:])  # new AA from known variant
                        if new_aa not in aa_list:
                            aa_list.append(new_aa)
                    else:
                        aa_list = self.amino_acids  # new AA can be any of the 20 standard AA's
            
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
            assert variant_amino_acid == s_traj[i+1][variant_position]

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
        v_traj, s_traj, y_traj = [], [], []
        v_traj.append('WT')
        y_traj.append(np.nan)
        s_traj.append(self.s_wt)
        accepted = 0
        add_epsilon = 0.0
        wt_prediction = None

        for iteration in range(self.num_iterations):
            self.de_step_counter = iteration

            if accepted == 0:
                prior_mutation_location = random.randint(0, len(self.s_wt))
            else:
                prior_mutation_location = int(re.findall(r"\d+", v_traj[-1])[0])
            
            prior_y = y_traj[-1]
            prior_sequence = s_traj[-1]

            new_var_seq = self.mutate_sequence(
                seq=prior_sequence,
                prev_mut_loc=prior_mutation_location
            )

            new_variant = new_var_seq[0][0]  # e.g., '17A'
            wt_pos = int(re.findall(r"\d+", new_variant)[0]) - 1
            new_full_variant = f"{self.s_wt[wt_pos]}{new_variant}"  # full variant name, e.g. 'F17A'
            new_sequence = new_var_seq[0][1]

            # encode and predict new sequence fitness
            if self.ml_or_hybrid == 'ml':
                if wt_prediction is None or wt_prediction == 'skip':
                    wt_prediction = 'skip'
                    while wt_prediction == 'skip':
                        rand_pos = random.randint(0, len(self.s_wt) - 1)
                        wt_mut = self.s_wt[rand_pos] + str(rand_pos) + self.s_wt[rand_pos]
                        logger.info(f"Trying to get WT fitness: {wt_mut}...")
                        wt_prediction = predict(
                            path=self.path,
                            model=self.model,
                            encoding=self.encoding,
                            variants=np.atleast_1d(wt_mut),
                            sequences=np.atleast_1d(self.s_wt),
                            no_fft=self.no_fft,
                            couplings_file=self.dca_encoder
                        )
                    wt_prediction = wt_prediction[0]
                
                if self.de_step_counter == 0:
                    logger.info(
                        f"Step {self.de_step_counter}: "
                        f"WT ({wt_mut}) --> {wt_prediction[0]:.3f} WT relative fitness: "
                        f"{wt_prediction[0] - wt_prediction[0] + add_epsilon:.3f}"
                    )
                    y_traj[0] = wt_prediction[0]
                
                predictions = predict(
                    path=self.path,
                    model=self.model,
                    encoding=self.encoding,
                    variants=np.atleast_1d(new_full_variant),
                    sequences=np.atleast_1d(new_sequence),
                    no_fft=self.no_fft,
                    couplings_file=self.dca_encoder
                )
                if predictions != "skip":
                    predictions = predictions[0]

            else:  # hybrid modeling and prediction
                if wt_prediction is None or wt_prediction == 'skip':
                    wt_prediction = 'skip'
                    while wt_prediction == 'skip':
                        rand_pos = random.randint(0, len(self.s_wt) - 1)
                        wt_mut = self.s_wt[rand_pos] + str(rand_pos + 1) + self.s_wt[rand_pos]
                        logger.info(f"Trying to get WT fitness: {wt_mut}...")
                        wt_prediction = predict_directed_evolution(
                            encoder=self.dca_encoder,
                            variant=wt_mut,
                            variant_sequence=self.s_wt,
                            hybrid_model_data_pkl=self.model
                        )
                if self.de_step_counter == 0:
                    logger.info(
                        f"Step {self.de_step_counter}: "
                        f"WT ({wt_mut}) --> {wt_prediction[0]:.3f} WT relative fitness: "
                        f"{wt_prediction[0] - wt_prediction[0] + add_epsilon:.3f}"
                    )
                    y_traj[0] = wt_prediction[0] - wt_prediction[0]
                
                predictions = predict_directed_evolution(
                    encoder=self.dca_encoder,
                    variant=new_full_variant,
                    variant_sequence=new_sequence,
                    hybrid_model_data_pkl=self.model
                )

            if predictions != 'skip':
                logger.info(
                    f"Step {self.de_step_counter + 1}: "
                    f"{new_full_variant} --> "
                    f"{predictions[0]:.3f} WT relative fitness: "
                    f"{predictions[0] - wt_prediction[0] + add_epsilon:.3f}"
                )
            else:  # skip if variant cannot be encoded
                logger.info(
                    f"Step {self.de_step_counter + 1}: "
                    f"{new_full_variant} --> {predictions}"
                )
                continue

            new_y = predictions[0] - wt_prediction[0] + add_epsilon
            new_var = new_full_variant  # Store full variant name (e.g., 'F17A')

            # probability function for trial sequence
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    boltz = np.exp(((new_y - prior_y) / self.temp), dtype=np.longdouble)
                    if self.negative:
                        boltz = np.exp((-(new_y - prior_y) / self.temp), dtype=np.longdouble)
                except OverflowError:
                    boltz = 1
            
            p = min(1, boltz)
            rand_var = random.random()
            if rand_var < p:  # Metropolis-Hastings acceptance
                v_traj.append(str(new_var))  # update variant trajectory with full name
                y_traj.append(new_y)
                s_traj.append(new_sequence)
                accepted += 1
                logger.info(f'Accepted variant {new_var} (current evolutionary trajectory: {v_traj})')
            else: 
                logger.info(f'Rejected variant {new_var} (current evolutionary trajectory: {v_traj})')

        self.assert_trajectory_sequences(v_traj, s_traj)

        return v_traj, s_traj, y_traj

    def run_de_trajectories(self):
        v_records = []
        s_records = []
        y_records = []
        for i in range(self.num_trajectories):
            self.traj_counter = i
            v_traj, s_traj, y_traj = self.in_silico_de()
            v_records.append(v_traj)
            s_records.append(s_traj)
            y_records.append(y_traj)

        return s_records, v_records, y_records

    def plot_trajectories(self):
        """
        Plots evolutionary trajectories and saves steps in CSV file.
        """
        s_records, v_records, y_records = self.run_de_trajectories()
        
        logger.info('Plotting evolution trajectories...')
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.locator_params(integer=True)
        y_records_ = []
        for i, fitness_array in enumerate(y_records):
            ax.plot(np.arange(1, len(fitness_array) + 1, 1), fitness_array,
                    '-o', alpha=0.7, markeredgecolor='black', label='EvoTraj' + str(i + 1))
            y_records_.append(fitness_array)
        
        label_x_y_name = []
        traj_max_len = 0
        for i, v_record in enumerate(v_records):
            for j, v in enumerate(v_record):
                if len(v_record) > traj_max_len:
                    traj_max_len = len(v_record)
                
                # Format full variant name (e.g. A123C) if short string (e.g. 123C) is passed
                v_name = v
                if v != 'WT' and v[0].isdigit():
                    pos_int = int(re.findall(r"\d+", v)[0]) - 1
                    v_name = f"{self.s_wt[pos_int]}{v}"

                if i == 0:  # j + 1 -> x-axis position shifted by 1
                    label_x_y_name.append(ax.text(j + 1, y_records_[i][j], v_name, size=7))
                else:
                    if v != 'WT':  # only plot 'WT' name once at i == 0
                        label_x_y_name.append(ax.text(j + 1, y_records_[i][j], v_name, size=7))
        
        adjust_text(label_x_y_name, only_move={'points': 'y', 'text': 'y'}, force_points=0.6)
        ax.legend()
        plt.xticks(np.arange(1, traj_max_len + 1, 1), np.arange(1, traj_max_len + 1, 1))

        plt.ylabel('Predicted fitness')
        plt.xlabel('Mutation trial steps')
        plt.tight_layout()
        fig_name = os.path.abspath(str(self.model) + '_DE_trajectories.png')
        plt.savefig(fig_name, dpi=500)
        plt.clf()
        plt.close('all')
        logger.info(f'Saved EvoTrajectory image as {fig_name}')

        evo_csv = os.path.abspath(os.path.join('EvoTraj', 'Trajectories.csv'))
        with open(evo_csv, 'w') as file:
            file.write('Trajectory;Variant;Sequence;Fitness\n')
            for i in range(self.num_trajectories):
                v_records_str = str(v_records[i])[1:-1].replace("'", "")
                s_records_str = str(s_records[i])[1:-1].replace("'", "")
                y_records_str = str(y_records[i])[1:-1]
                file.write(f'{i+1};{v_records_str};{s_records_str};{y_records_str}\n')
        logger.info(f'Saved EvoTrajectory CSV as {evo_csv}')
