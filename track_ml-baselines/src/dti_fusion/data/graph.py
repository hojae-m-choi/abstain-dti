"""RDKit molecule -> dense graph tensors for the Graph Transformer.

Each molecule becomes
  atom_feat  [N, F] int   categorical atom features (index 0 is reserved for padding)
  bond       [N, N] int   bond type between atoms (0 = no bond)
  spd        [N, N] int   shortest-path distance in bonds (unreachable -> -1)
Hydrogens stay implicit.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem

_HYBRID = {
    Chem.rdchem.HybridizationType.SP: 1,
    Chem.rdchem.HybridizationType.SP2: 2,
    Chem.rdchem.HybridizationType.SP3: 3,
    Chem.rdchem.HybridizationType.SP3D: 4,
    Chem.rdchem.HybridizationType.SP3D2: 5,
}
_CHIRAL = {
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED: 0,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW: 1,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: 2,
}
_BOND = {
    Chem.rdchem.BondType.SINGLE: 1,
    Chem.rdchem.BondType.DOUBLE: 2,
    Chem.rdchem.BondType.TRIPLE: 3,
    Chem.rdchem.BondType.AROMATIC: 4,
}
NUM_BOND_TYPES = 6  # 0 none, 1-4 above, 5 other

# (name, vocabulary size excluding the padding slot)
ATOM_FEATURES = (
    ("atomic_num", 101),  # 1..100, 101 = other
    ("degree", 7),  # 0..5, 6 = more
    ("formal_charge", 5),  # -2..+2 (clipped)
    ("num_hs", 5),  # 0..4
    ("hybridization", 6),  # 0 other, 1..5
    ("aromatic", 2),
    ("in_ring", 2),
    ("chirality", 4),  # 0 none, 1 CW, 2 CCW, 3 other
)


def _atom_row(atom: Chem.Atom) -> list[int]:
    values = [
        min(atom.GetAtomicNum(), 101) - 1,
        min(atom.GetDegree(), 6),
        int(np.clip(atom.GetFormalCharge(), -2, 2)) + 2,
        min(atom.GetTotalNumHs(), 4),
        _HYBRID.get(atom.GetHybridization(), 0),
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
        _CHIRAL.get(atom.GetChiralTag(), 3),
    ]
    return [v + 1 for v in values]  # shift so 0 means padding


def smiles_to_graph(smiles: str) -> dict[str, np.ndarray]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"cannot parse SMILES: {smiles}")
    atom_feat = np.array([_atom_row(a) for a in mol.GetAtoms()], dtype=np.int64)
    n = mol.GetNumAtoms()
    bond = np.zeros((n, n), dtype=np.int64)
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        t = _BOND.get(b.GetBondType(), 5)
        bond[i, j] = bond[j, i] = t
    dist = Chem.GetDistanceMatrix(mol)  # disconnected fragments get 1e8
    spd = np.where(dist > 1e6, -1, dist).astype(np.int64)
    return {"atom_feat": atom_feat, "bond": bond, "spd": spd}
