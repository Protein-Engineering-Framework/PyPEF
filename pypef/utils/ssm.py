# PyPEF - Pythonic Protein Engineering Framework
# https://github.com/niklases/PyPEF


import logging
logger = logging.getLogger('pypef.utils.ssm')

import os
from typing import Literal, Union
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np

from pypef.hybrid.hybrid_model import get_model_and_type
from pypef.llm.inference import inference
from pypef.dca.gremlin_inference import GREMLIN


class SSM:
    def __init__(
            self,
            wt_seq: str,
            aas: str = "ARNDCQEGHILKMFPSTWYV",
            model: Literal["dca", "esm", "prosst"] | None = None,
            pdb: str | os.PathLike | None = None,
            params_file: str | os.PathLike | None = None
    ):
        self.wt_seq = wt_seq
        self.aas = aas
        if model is None:
            model = "dca"
        self.model = model
        self.pdb = pdb
        if params_file is not None:
            gremlin, m_type = get_model_and_type(params_file)
            if m_type != "GREMLIN":
                raise RuntimeError("Model is not a GREMLIN model!")
        else:
            gremlin = None
        self.gremlin = gremlin
        self.get_data()
        self.predict()
        self.plot()

    def get_data(self):
        """
        Function to plot all predicted 19 amino acid substitution 
        effects at all predictable WT/input sequence positions; e.g.: 
        M1A, M1C, M1E, ..., D2A, D2C, D2E, ..., ..., T300V, T300W, T300Y
        """
        self.variantss, self.variant_sequencess = [], []
        for i, aa_wt in enumerate(self.wt_seq):
            variants, variant_sequences = [], []
            for aa_sub in self.aas:
                variant = aa_wt + str(i + 1) + aa_sub
                variant_sequence = self.wt_seq[:i] + aa_sub + self.wt_seq[i + 1:]
                variants.append(variant)
                variant_sequences.append(variant_sequence)
            self.variantss.append(variants)
            self.variant_sequencess.append(variant_sequences)

    def predict(self):
        if self.model == "dca" and self.gremlin is not None:
            logger.info("Predicting all SSM effects using the unsupervised GREMLIN model...")
            wt_score = self.gremlin.get_wt_score()
        self.scoress = []
        if self.model == "dca" and self.gremlin is not None:
            for seqs in tqdm(self.variant_sequencess, desc="Predicting seq. pos. substitution effects"):
                self.scoress.append(self.gremlin.get_scores(seqs) - wt_score)
        elif self.model in ["esm", "prosst"]:
            self.scoress = inference(
                        np.array(self.variant_sequencess).flatten(), llm=self.model, pdb_file=self.pdb, wt_seq=self.wt_seq
            ).numpy()
            logger.info(f"Reshaping flat array of shape {np.shape(self.scoress)} "
                        f"to SSM shape {np.shape(self.variant_sequencess)}...")
            self.scoress = self.scoress.reshape(np.shape(self.variant_sequencess))
        else:
            raise RuntimeError("Unknown modeling option (choose between --llm esm or --llm prosst)!")

    def plot(self):
        _fig, ax = plt.subplots(figsize=(2 * len(self.wt_seq) / len(self.aas), 3))
        ax.imshow(np.array(self.scoress).T)
        for i_vss, vss in enumerate(self.scoress):
            for i_vs, vs in enumerate(vss):
                ax.text(
                    i_vss, i_vs, 
                    f'{self.variantss[i_vss][i_vs]}\n{round(vs, 1):.1f}', 
                    size=1.5, va='center', ha='center'
                )
        ax.set_xticks(
            range(len(self.wt_seq)), 
            [f'{aa}{i + 1}' for i, aa in enumerate(self.wt_seq)], 
            size=6, rotation=90
        )
        ax.set_yticks(range(len(self.aas)), self.aas, size=6)
        plt.tight_layout()
        plt.savefig('SSM_landscape.png', dpi=500)
        plt.clf()
        plt.close('all')
        pd.DataFrame(
            {
                'Variant': np.array(self.variantss).flatten(),
                'Sequence': np.array(self.variant_sequencess).flatten(),
                'Variant_Score': np.array(self.scoress).flatten()
            }
        ).to_csv('SSM_landscape.csv', sep=',')
        logger.info(f"Saved SSM landscape as {os.path.abspath('SSM_landscape.png')} "
                    f"and CSV data as {os.path.abspath('SSM_landscape.csv')}...")
