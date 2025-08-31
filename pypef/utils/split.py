# PyPEF - Pythonic Protein Engineering Framework
# https://github.com/niklases/PyPEF


import pandas as pd
from os import PathLike, path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

import logging
logger = logging.getLogger('pypef.utils.split')

from typing import Union
FilePath = Union[str, "PathLike[str]"]


class DatasetSplitter:
    def __init__(
            self, 
            df_or_csv_file: FilePath | pd.DataFrame, 
            n_cv: int | None = None,
            mutation_column: str | None = None,
            mutation_separator: str | None = None,
            csv_separator: str | None = None
    ):
        self.df_or_csv_file = df_or_csv_file
        self.mutation_column = mutation_column
        if csv_separator is None:
            csv_separator = ','
        if mutation_separator is None:
            mutation_separator = '/'
        self.mutation_separator = mutation_separator
        self.csv_separator = csv_separator
        if n_cv is None:
            n_cv = 5
        self.n_cv = n_cv
        if type(df_or_csv_file) == pd.DataFrame:
            self.df = self.df_or_csv_file
            self.fig_path = path.abspath('CV_split_pos_aa_distr.png')
        else:
            self.df = pd.read_csv(self.df_or_csv_file, sep=self.csv_separator)
            self.fig_path = path.abspath(path.splitext(path.basename(
                self.df_or_csv_file))[0] + '_pos_aa_distr.png')
        logger.info(f'Dataframe size: {self.df.shape[0]}')
        self.random_splits_train_indices_combined, self.random_splits_test_indices_combined = None, None
        self.modulo_splits_train_indices_combined, self.modulo_splits_test_indices_combined = None, None
        self.cont_splits_train_indices_combined, self.cont_splits_test_indices_combined = None, None
        self.random_splits_train_indices_combined_multi = None
        self.random_splits_test_indices_combined_multi = None
        self.order_by_pos()
        self.split_random()
        self.split_modulo()
        self.split_continuous()
        self.split_random_single_and_multi()
    
    def order_by_pos(self):
        if self.mutation_column is None:
            variants = self.df.iloc[:, 0].to_list()
        else:
            variants = self.df[self.mutation_column].to_list()
        single_mut_idxs = []
        for i, variant in enumerate(variants):
            if not self.mutation_separator in variant:
                single_mut_idxs.append(i)
        if single_mut_idxs:   
            self.df_singles = self.df.loc[single_mut_idxs, :]
            self.df_singles.index.name = 'old_index'
            self.df_singles.reset_index(inplace=True)  # Keeping the index copy of the full old df
            self.df_singles['old_index'] = self.df_singles.pop('old_index')  # Move to end (last column)
            if self.df_singles.size != self.df.size:
                logger.info(  #(TODO): For now, removing double or higher mutated variants (keep in future?!)..
                    f'Removed {self.df.shape[0] - self.df_singles.shape[0]} multimutated variants '
                    f'from dataframe (for plotting and modulo and continuous cross-validation splitting)... '
                    f'new dataframe size: {self.df_singles.shape[0]}'
                )
        if self.mutation_column is None:
            variants = self.df_singles.iloc[:, 0].to_list()
        else:
            variants = self.df_singles[self.mutation_column].to_list()
        self.df_singles['variant_pos'] = [int(v[1:-1]) for v in variants]
        self.df_singles['substitutions'] = [v[-1] for v in variants]
        self.df_singles.sort_values(['variant_pos', 'substitutions'], ascending=[True, True], inplace=True)
        self.min_pos, self.max_pos = self.df_singles['variant_pos'].to_numpy()[0], self.df_singles['variant_pos'].to_numpy()[-1]

    def get_old_index(self, new_indices: list):
        if isinstance(new_indices[0], (list, np.ndarray)):  # 2D array/list
            temp = []
            for inner in new_indices:
                temp.append(self.df_singles.loc[inner]['old_index'].tolist())
            return temp
        else:  # 1D array/list
            return self.df_singles.loc[new_indices]['old_index'].tolist()

    def split_random(self):
        self.random_splits_train_indices_combined = []
        self.random_splits_test_indices_combined = []
        kf = KFold(n_splits=self.n_cv, shuffle=True, random_state=42)
        for i_train, i_test in kf.split(range(self.df_singles.shape[0])):
            self.random_splits_train_indices_combined.append(i_train)
            self.random_splits_test_indices_combined.append(i_test)

    def split_modulo(self):
        """
        Likely inhomogeneous shape as not all protein backbone positions 
        are necessarily equally frequent in the dataset.
        """
        modulo_splits = [[] for _ in range(self.n_cv)]
        for i_v, v_pos in enumerate(self.df_singles['variant_pos'].to_numpy()):
            for i in range(self.n_cv):
                if v_pos % self.n_cv == i:
                    modulo_splits[i].append(i_v)
        modulo_train_splits = []
        self.modulo_splits_train_indices_combined = []
        for i, _split in enumerate(modulo_splits):
            modulo_train_splits.append([split for j, split in enumerate(modulo_splits) if i != j])
        for splits in modulo_train_splits:
                temp = []
                for split in splits:
                    temp += split
                self.modulo_splits_train_indices_combined.append(temp)
        self.modulo_splits_test_indices_combined = [[] for _ in range(self.n_cv)]
        for i_ts, train_split in enumerate(self.modulo_splits_train_indices_combined):
            for i in range(self.df_singles.shape[0]):
                if i not in train_split:
                    self.modulo_splits_test_indices_combined[i_ts].append(i)

    def split_continuous(self):
        """
        Similar to kf = KFold(n_splits=self.n_cv, shuffle=False) when 
        ordering variants prior to k-fold cross-validation.
        """
        cont_poses = np.array_split(np.array(range(self.min_pos, self.max_pos + 1)), self.n_cv)
        cont_splits = [[] for _ in range(self.n_cv)]
        for i_p, poses in enumerate(cont_poses):
            for i, pos in enumerate(self.df_singles['variant_pos'].to_numpy()):
                if pos in poses:
                    cont_splits[i_p].append(i)
        cont_train_splits = []
        self.cont_splits_train_indices_combined = []
        for i, _split in enumerate(cont_splits):
            cont_train_splits.append([split for j, split in enumerate(cont_splits) if i != j])
        for splits in cont_train_splits:
                temp = []
                for split in splits:
                    temp += split
                self.cont_splits_train_indices_combined.append(temp)
        self.cont_splits_test_indices_combined = [[] for _ in range(self.n_cv)]
        for i_ts, train_split in enumerate(self.cont_splits_train_indices_combined):
            for i in range(self.df_singles.shape[0]):
                if i not in train_split:
                    self.cont_splits_test_indices_combined[i_ts].append(i)
    
    def split_random_single_and_multi(self):
        self.random_splits_train_indices_combined_multi = []
        self.random_splits_test_indices_combined_multi = []
        kf = KFold(n_splits=self.n_cv, shuffle=True, random_state=42)
        for i_train, i_test in kf.split(range(self.df.shape[0])):
            self.random_splits_train_indices_combined_multi.append(i_train)
            self.random_splits_test_indices_combined_multi.append(i_test)

    def print_shapes(self):
        """
        Also gets inhomogeneous shapes (using for loop on sublists of nested lists instead 
        of np.shape() on entire nested list). 
        """
        random_shape_train = [np.shape(k) for k in self.random_splits_train_indices_combined]
        random_shape_test = [np.shape(k) for k in self.random_splits_test_indices_combined]
        logger.info(f'Random train --> test split shapes: {random_shape_train} --> {random_shape_test}')
        
        modulo_shape_train = [np.shape(k) for k in self.modulo_splits_train_indices_combined]
        modulo_shape_test = [np.shape(k) for k in self.modulo_splits_test_indices_combined]
        logger.info(f'Modulo train --> test split shapes: {modulo_shape_train} --> {modulo_shape_test}')
        
        cont_shape_train = [np.shape(k) for k in self.cont_splits_train_indices_combined]
        cont_shape_test = [np.shape(k) for k in self.cont_splits_test_indices_combined]
        logger.info(f'Continuous train --> test split shapes: {cont_shape_train} --> {cont_shape_test}')

    def _get_zero_counts(self) -> dict:
        all_poses = np.asarray(range(self.min_pos, self.max_pos + 1))
        zero_counts = np.zeros_like(all_poses)
        return dict(zip(all_poses, zero_counts))
    
    def _get_distribution(self, indices):
        df_fold = self.df_singles.iloc[indices, :]
        un, c = np.unique(df_fold['variant_pos'].to_numpy(), return_counts=True)
        zc = self._get_zero_counts()
        zc.update(dict(zip(un, c)))
        return list(zc.keys()), list(zc.values())
    
    def get_single_sub_df(self):
        return self.df_singles

    def get_all_split_indices(self, old_index: bool = True):
        if old_index:
            return [
                [self.get_old_index(self.random_splits_train_indices_combined), 
                 self.get_old_index(self.random_splits_test_indices_combined)],
                [self.get_old_index(self.modulo_splits_train_indices_combined), 
                 self.get_old_index(self.modulo_splits_test_indices_combined)],
                [self.get_old_index(self.cont_splits_train_indices_combined), 
                 self.get_old_index(self.cont_splits_test_indices_combined)]
            ]
        else:
            return [
                [self.random_splits_train_indices_combined, self.random_splits_test_indices_combined],
                [self.modulo_splits_train_indices_combined, self.modulo_splits_test_indices_combined],
                [self.cont_splits_train_indices_combined, self.cont_splits_test_indices_combined]
            ]
    
    def get_random_single_multi_split_indices(self):
        return [
            self.random_splits_train_indices_combined_multi,
            self.random_splits_test_indices_combined_multi
        ]
    
    def _get_df_split_data(self, combined_train_indices, combined_test_indices, target_df=None):
        if target_df is None:
            target_df = self.df_singles
        train_split_data, test_split_data = [], []
        for train_split, test_split in zip(combined_train_indices, combined_test_indices):
            train_split_data.append(
                target_df.iloc[train_split, :].reset_index(drop=True)
            )
            test_split_data.append(
                target_df.iloc[test_split, :].reset_index(drop=True)
            )
        return train_split_data, test_split_data
    
    def get_random_df_split_data(self, include_multis: bool = False):
        if include_multis:
            target_df = self.df
            combined_train_indices = self.random_splits_train_indices_combined_multi
            combined_test_indices = self.random_splits_test_indices_combined_multi
        else:
            target_df = self.df_singles
            combined_train_indices = self.random_splits_train_indices_combined
            combined_test_indices = self.random_splits_test_indices_combined
        return self._get_df_split_data(
            combined_train_indices, 
            combined_test_indices,
            target_df
        )

    def get_modulo_df_split_data(self):
        return self._get_df_split_data(
            self.modulo_splits_train_indices_combined, 
            self.modulo_splits_test_indices_combined
        )

    def get_continuous_df_split_data(self):
        return self._get_df_split_data(
            self.cont_splits_train_indices_combined, 
            self.cont_splits_test_indices_combined
        )

    def plot_distributions(self):
        fig, axs = plt.subplots(
            nrows=4, ncols=self.n_cv,  
            constrained_layout=True
        )
        logger.info("Plotting distributions...")
        fig.set_figwidth(30)
        fig.set_figheight(10)
        poses, counts = self._get_distribution(sorted(list(self.df_singles.index)))
        for i in range(self.n_cv):
            if i == self.n_cv // 2:
                axs[0, i].set_title("All data")
                axs[0, i].plot(poses, counts, color='black')
                axs[0, i].set_ylim(0, 20)
                axs[0, i].set_xlim(self.min_pos - 4, self.max_pos + 4)
                axs[0, i].set_ylabel(f"# Amino acids")
            else:
                fig.delaxes(axs[0, i])
        for i_category, (train_indices, test_indices) in enumerate(self.get_all_split_indices(old_index=False)):
            category = ["Random", "Modulo", "Continuous"][i_category]
            for i_split in range(self.n_cv):
                pos_train, counts_train = self._get_distribution(train_indices[i_split])
                pos_test, counts_test = self._get_distribution(test_indices[i_split])
                axs[i_category + 1, i_split].plot(
                    pos_train, counts_train, marker="o", linestyle="--", markersize=3, linewidth=0.5
                )
                axs[i_category + 1, i_split].plot(
                    pos_test, counts_test, marker="o", linestyle="--", markersize=3, linewidth=0.5
                )
                xticks = list(axs[i_category + 1, i_split].get_xticks())
                xticks = xticks[1:-1]
                if 0 in xticks:
                    xticks.remove(0)
                xticks.append(self.min_pos) 
                xticks.append(self.max_pos)
                xticks = sorted(xticks)
                if len(xticks) >= 3:
                    if (xticks[-1] - xticks[-2]) < 0.5 * (xticks[2] - xticks[1]):
                        xticks.pop()
                    if (xticks[1] - xticks[0]) < 0.5 * (xticks[2] - xticks[1]):
                        xticks.pop(0)
                axs[i_category + 1, i_split].set_xticks(xticks)
                if i_category == 0:
                    axs[i_category + 1, i_split].set_title(f"Split {i_split + 1}")
                if i_category == 2:
                    axs[i_category + 1, i_split].set_xlabel(f"Residue position")
                if i_split == 0:
                    axs[i_category + 1, i_split].set_ylabel(f"# Amino acids")
                if i_split == self.n_cv // 2:
                    axs[i_category + 1, i_split].set_title(category)
                axs[i_category + 1, i_split].set_ylim(0, 20)
                axs[i_category + 1, i_split].set_xlim(self.min_pos - 4, self.max_pos + 4)
        axs[0, self.n_cv // 2].set_xticks(xticks)
        
        plt.savefig(self.fig_path, dpi=300)
        logger.info(f"Saved figure as {self.fig_path}.")
        plt.close(fig)
