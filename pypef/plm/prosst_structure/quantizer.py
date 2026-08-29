# PyPEF - Pythonic Protein Engineering Framework
# https://github.com/niklases/PyPEF

import biotite.structure as struc
import biotite.structure.io as strucio
from biotite.sequence import ProteinSequence
import math
import numpy as np
import os
import torch
import torch.nn.functional as F
from pathlib import Path
from torch_geometric.data import Batch, Data

from pypef.plm.prosst_structure.encoder.gvp import AutoGraphEncoder
from pypef.plm.prosst_structure.scatter import scatter_mean
from pypef.utils.helpers import get_device

import logging
logger = logging.getLogger('pypef.llm.prosst_structure.quantizer')


@torch.jit.script
def _normalize(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.nan_to_num(torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def _rbf(D: torch.Tensor, D_min: float = 0.0, D_max: float = 20.0, D_count: int = 16) -> torch.Tensor:
    D_mu = torch.linspace(D_min, D_max, D_count, device=D.device).view(1, -1)
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    return torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)


@torch.jit.script
def _orientations(X_ca):
    forward = _normalize(X_ca[1:] - X_ca[:-1])
    backward = _normalize(X_ca[:-1] - X_ca[1:])
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)


@torch.jit.script
def _sidechains(X):
    n, origin, c = X[:, 0], X[:, 1], X[:, 2]
    c, n = _normalize(c - origin), _normalize(n - origin)
    bisector = _normalize(c + n)
    perp = _normalize(torch.cross(c, n, dim=1))
    vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
    return vec

def _positional_embeddings(edge_index, num_embeddings=16):
    d = edge_index[0] - edge_index[1]
    frequency = torch.exp(torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=edge_index.device)
                          * -(np.log(10000.0) / num_embeddings))
    angles = d.unsqueeze(-1) * frequency
    return torch.cat((torch.cos(angles), torch.sin(angles)), -1)


class PdbQuantizer:
    def __init__(
        self,
        max_distance=10,
        subgraph_depth=40,
        subgraph_interval=1,
        model_path=None,
        cluster_dir=None,
        cluster_model=None,
        device=None,
        verbose: bool = True
    ) -> None:
        self.max_distance = max_distance
        self.subgraph_depth = subgraph_depth or 40
        self.subgraph_interval = subgraph_interval
        self.device = device or get_device()
        self.verbose = verbose
        
        if model_path is None:
            name = "AE_CPU.pt" if self.device == 'cpu' else "AE.pt"
            self.model_path = str(Path(__file__).parent / "static" / name)
        else:
            self.model_path = model_path
        
        self.cluster_dir = cluster_dir or str(Path(__file__).parent / "static")
        self.cluster_files = cluster_model or ["2048_kmeans_cluster_centers.npy"]

        self.model = AutoGraphEncoder(
            node_in_dim=(20, 3), 
            node_h_dim=(256, 32),
            edge_in_dim=(32, 1), 
            edge_h_dim=(64, 2), 
            num_layers=6
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
        self.model.eval()

        self.centers = {}
        for cf in self.cluster_files:
            p = os.path.join(self.cluster_dir, cf)
            name = Path(cf).stem
            self.centers[name] = torch.from_numpy(np.load(p)).to(self.device).float()

    @torch.no_grad()
    def __call__(self, pdb_file, return_residue_seq=False):
        struct = strucio.load_structure(pdb_file, model=1)
        struct = struct[struc.filter_amino_acids(struct)]
        all_coords = []
        residue_indices = sorted(list(set(struct.res_id)))
        valid_res_names = []
        for res_id in residue_indices:
            res_atoms = struct[struct.res_id == res_id]
            n = res_atoms[res_atoms.atom_name == "N"].coord
            ca = res_atoms[res_atoms.atom_name == "CA"].coord
            c = res_atoms[res_atoms.atom_name == "C"].coord
            
            if len(n) > 0 and len(ca) > 0 and len(c) > 0:
                all_coords.append([n[0], ca[0], c[0]])
                valid_res_names.append(res_atoms.res_name[0])
        
        coords = torch.as_tensor(
            np.array(all_coords), 
            dtype=torch.float32, 
            device=self.device
        )

        ca_coords = coords[:, 1]
        L = ca_coords.size(0)
        
        node_s_full = torch.zeros(L, 20, device=self.device)
        node_v_full = torch.cat([_orientations(ca_coords), _sidechains(coords).unsqueeze(-2)], dim=-2)
        dist_matrix = torch.cdist(ca_coords, ca_coords)

        subgraph_list = []
        indices_range = range(0, L, self.subgraph_interval)
        
        for i in indices_range:
            dist_row = dist_matrix[i]
            
            _, top_k = torch.topk(dist_row, k=min(self.subgraph_depth, L), largest=False)
            neighbor_idx = top_k[dist_row[top_k] < self.max_distance].sort().values

            sub_edge_index_local = (dist_matrix[neighbor_idx][:, neighbor_idx] < self.max_distance).nonzero().t()
            mask = sub_edge_index_local[0] != sub_edge_index_local[1]
            sub_edge_index_local = sub_edge_index_local[:, mask]
            
            orig_src = neighbor_idx[sub_edge_index_local[0]]
            orig_dst = neighbor_idx[sub_edge_index_local[1]]
            
            edge_vecs = ca_coords[orig_src] - ca_coords[orig_dst]
            sub_edge_s = torch.cat([
                _rbf(edge_vecs.norm(dim=-1), D_count=16), 
                _positional_embeddings(torch.stack([orig_src, orig_dst]))
            ], dim=-1)
            sub_edge_v = _normalize(edge_vecs).unsqueeze(-2)

            subgraph_list.append(Data(
                node_s=node_s_full[neighbor_idx], 
                node_v=node_v_full[neighbor_idx],
                edge_index=sub_edge_index_local, 
                edge_s=sub_edge_s, 
                edge_v=sub_edge_v
            ))

        batch = Batch.from_data_list(subgraph_list).to(self.device)
        node_embeddings = self.model.get_embedding(
            (batch.node_s, batch.node_v), 
            batch.edge_index, 
            (batch.edge_s, batch.edge_v)
        )
        
        # graph_embeddings = scatter_mean_simple(node_embeddings, batch.batch, batch.num_graphs)
        graph_embeddings = scatter_mean(node_embeddings, batch.batch, dim=0)
        
        norm_embeds = F.normalize(graph_embeddings, p=2, dim=1)

        center_key = list(self.centers.keys())[0]
        centers = self.centers[center_key]
        
        tokens = torch.cdist(norm_embeds, centers, p=2).argmin(dim=1).cpu().tolist()

        if return_residue_seq:
            residues = struc.get_residues(struct)[1]
            aa_seq = "".join([ProteinSequence.convert_letter_3to1(r) for r in residues])
            return aa_seq, tokens
        return tokens
    