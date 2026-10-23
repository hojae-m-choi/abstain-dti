"""Fast unit tests: no downloads, random tensors only."""

import numpy as np
import pandas as pd
import pytest
import torch

from dti_fusion.config import load_config
from dti_fusion.data.dataset import Collator
from dti_fusion.data.featurize import TokenStore, _window_starts
from dti_fusion.data.graph import smiles_to_graph
from dti_fusion.data.splits import make_split
from dti_fusion.metrics import concordance_index, regression_metrics
from dti_fusion.models.model import DTIModel

SMILES = ["CCO", "c1ccccc1O", "CC(=O)Nc1ccc(O)cc1", "C1CCCCC1N", "O=C(O)c1ccccc1", "CCN(CC)CC"]


def _toy_df(n_drugs=6, n_targets=10):
    rows = [(f"d{i}", f"t{j}", SMILES[i % len(SMILES)], float(i + j)) for i in range(n_drugs) for j in range(n_targets)]
    return pd.DataFrame(rows, columns=["drug_key", "target_key", "smiles", "y"])


def test_graph_featurization_shapes():
    g = smiles_to_graph("CC(=O)Nc1ccc(O)cc1.Cl")  # includes a disconnected fragment
    n = g["atom_feat"].shape[0]
    assert g["bond"].shape == (n, n) and g["spd"].shape == (n, n)
    assert (g["atom_feat"] > 0).all()  # 0 reserved for padding
    assert (g["spd"] == -1).any()  # fragment unreachable
    assert (np.diag(g["spd"]) == 0).all()


@pytest.mark.parametrize("kind", ["random", "cold_drug", "cold_target", "cold_both", "scaffold"])
def test_splits_disjoint_and_cold(kind):
    df = _toy_df(20, 20)
    df["smiles"] = [SMILES[i % 6] + "C" * (i % 7) for i in range(len(df))]
    split = make_split(df, kind, (0.6, 0.2, 0.2), seed=1)
    all_idx = np.concatenate(list(split.values()))
    assert len(all_idx) == len(set(all_idx))
    train, test = df.iloc[split["train"]], df.iloc[split["test"]]
    if kind in ("cold_drug", "cold_both"):
        assert not set(train.drug_key) & set(test.drug_key)
    if kind in ("cold_target", "cold_both"):
        assert not set(train.target_key) & set(test.target_key)
    if kind == "scaffold":
        assert not set(train.smiles) & set(test.smiles)


def test_window_starts_cover_sequence():
    for length in (100, 1022, 1023, 2549, 4128):
        starts = _window_starts(length, 1022, 256)
        covered = np.zeros(length, bool)
        for s in starts:
            covered[s : s + 1022] = True
        assert covered.all() and starts[-1] + 1022 >= length


def test_token_store_roundtrip(tmp_path):
    items = {"a": np.random.randn(5, 8).astype(np.float32), "b": np.random.randn(3, 8).astype(np.float32)}
    store = TokenStore.write(tmp_path, items)
    reopened = TokenStore(tmp_path)
    np.testing.assert_allclose(reopened.get("b"), items["b"], atol=1e-2)
    assert reopened.dim == 8 and "a" in reopened and store.get("a").shape == (5, 8)


def test_concordance_index():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert concordance_index(y, y) == 1.0
    assert concordance_index(y, -y) == 0.0
    assert concordance_index(y, np.zeros(4)) == 0.5
    m = regression_metrics(y, y + 0.1, binary_threshold=2.5)
    assert m["auroc"] == 1.0 and abs(m["rmse"] - 0.1) < 1e-9


def _batch(drug_mode, protein_dim=32, token_dim=24):
    torch.manual_seed(0)
    proteins = [torch.randn(L, protein_dim) for L in (40, 17, 25)]
    if drug_mode == "graph":
        drugs = [smiles_to_graph(s) for s in SMILES[:3]]
    else:
        drugs = [torch.randn(L, token_dim) for L in (9, 4, 12)]
    collate = Collator(drug_mode)
    return collate(list(zip(drugs, proteins, [5.0, 6.0, 7.5])))


@pytest.mark.parametrize("encoder,mode", [("graph", "cross"), ("graph", "joint"), ("chemberta", "cross")])
def test_model_forward_backward(encoder, mode):
    cfg = load_config(overrides=[
        f"drug.encoder={encoder}", f"fusion.mode={mode}", "fusion.d_model=32", "fusion.n_heads=4",
        "drug.graph.d_model=32", "drug.graph.n_heads=4", "drug.graph.n_layers=2", "fusion.n_layers=1",
    ])
    batch = _batch("graph" if encoder == "graph" else "tokens")
    model = DTIModel(cfg, protein_dim=32, drug_token_dim=24)
    pred, attn = model(batch, return_attention=True)
    assert pred.shape == (3,) and torch.isfinite(pred).all()
    # padded protein positions get zero attention
    assert torch.all(attn["protein_attn"][1, 17:] == 0)
    pred.sum().backward()
    assert all(p.grad is not None for n, p in model.named_parameters() if p.requires_grad and "type_embedding" not in n)


def test_padding_invariance():
    """Extra padding in the batch must not change a sample's prediction."""
    cfg = load_config(overrides=["fusion.d_model=32", "fusion.n_heads=4", "drug.graph.d_model=32",
                                 "drug.graph.n_heads=4", "drug.graph.n_layers=2", "fusion.n_layers=1"])
    model = DTIModel(cfg, protein_dim=32).eval()
    g = smiles_to_graph("CCO")
    big = smiles_to_graph("CC(=O)Nc1ccc(O)cc1")
    p_small, p_big = torch.randn(10, 32), torch.randn(50, 32)
    collate = Collator("graph")
    alone = model(collate([(g, p_small, 0.0)]))
    padded = model(collate([(g, p_small, 0.0), (big, p_big, 0.0)]))
    torch.testing.assert_close(alone[0], padded[0], atol=1e-5, rtol=1e-4)


def test_config_overrides():
    cfg = load_config(overrides=["drug.encoder=chemberta", "data.fractions=[0.7,0.15,0.15]", "train.lr=1e-3"])
    assert cfg.drug.encoder == "chemberta" and cfg.data.fractions == (0.7, 0.15, 0.15) and cfg.train.lr == 1e-3
    with pytest.raises(KeyError):
        load_config(overrides=["fusion.nope=1"])


def test_structural_bias_matches_embedding_lookup():
    """The one-hot matmul bias must equal plain nn.Embedding lookups (the slow-on-MPS reference)."""
    from dti_fusion.models.graph_transformer import GraphTransformerEncoder

    enc = GraphTransformerEncoder(d_model=32, n_layers=1, n_heads=4, max_dist=3)
    torch.nn.init.normal_(enc.spd_bias.weight)
    torch.nn.init.normal_(enc.bond_bias.weight)
    torch.nn.init.normal_(enc.graph_token_bias)
    batch = Collator("graph")([(smiles_to_graph(s), torch.zeros(1, 4), 0.0) for s in ("CCO.Cl", "c1ccccc1CCCCCN")])
    bond, spd = batch["drug"]["bond"], batch["drug"]["spd"]
    bias = enc._structural_bias(bond, spd)

    bucket = torch.where(spd < 0, enc.max_dist + 2, spd.clamp(max=enc.max_dist + 1))
    ref = torch.zeros(bias.shape[0], spd.shape[1] + 1, spd.shape[1] + 1, 4)
    ref[:, 1:, 1:] = enc.spd_bias(bucket) + enc.bond_bias(bond)
    ref[:, 0, 1:] = enc.graph_token_bias[0]
    ref[:, 1:, 0] = enc.graph_token_bias[1]
    torch.testing.assert_close(bias, ref.permute(0, 3, 1, 2))
