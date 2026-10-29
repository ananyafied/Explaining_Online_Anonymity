"""Reload the step-1 models and rebuild the functions that score with them.

Every `score_*` function here reproduces what step 1 did to produce `outputs/step1_fit/predictions/<model>.csv`.
`verify_reload` checks that, so an explanation is never computed for a model that differs from the
one that was evaluated in step 1.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.sparse.linalg import splu

from . import step1_models as M
from .data import Data

DEVICE = torch.device("cpu")  # explanations run on CPU: small models, and Captum/PyG behave the same everywhere

STEP1_DIR = lambda D: D.root / "outputs" / "step1_fit"  # noqa: E731

TREE_MODELS = ["rf", "xgboost", "lightgbm"]
TABULAR_CLASSIFIERS = ["rf", "xgboost", "lightgbm", "svm_rbf", "svm_linear", "pca_logreg", "nmf_logreg", "svd_logreg"]
TABULAR_ANOMALY = ["isolation_forest", "lof", "one_class_svm", "autoencoder_dense"]
TABULAR_MODELS = TABULAR_CLASSIFIERS + TABULAR_ANOMALY
SEQ_MODELS = ["cnn1d", "rnn", "lstm", "autoencoder_lstm"]
TEXT_MODELS = ["transformer", "word2vec_logreg", "doc2vec_logreg"]
GNN_MODELS = {"gcn": "gcn", "gat": "gat", "graphsage": "sage", "rgcn": "rgcn"}
EMBED_MODELS = ["node2vec_logreg", "deepwalk_logreg"]
PROPAGATION_MODELS = ["label_spreading", "belief_propagation"]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ------------------------------------------------------------------ loading
def load_model(name: str, D: Data):
    """Load one step-1 model from outputs/step1_fit/models/."""
    d = STEP1_DIR(D) / "models"
    if name in GNN_MODELS:
        m = M.make_gnn(GNN_MODELS[name], len(D.FEATURES))
    elif name == "cnn1d":
        m = M.SeqCNN()
    elif name in ("rnn", "lstm"):
        m = M.SeqRNN(name)
    elif name == "autoencoder_lstm":
        m = M.LSTMAutoencoder()
    elif name == "autoencoder_dense":
        m = M.DenseAE(len(D.FEATURES))
    elif name == "transformer":
        ex = joblib.load(d / "transformer_extra.joblib")
        m = M.PostTransformer(len(ex["vocab"]), ex["max_len"])
    elif name == "multimodal_fusion":
        m = M.make_fusion(len(D.FEATURES), 64)  # 64 = transformer d_model
    elif name in ("word2vec_logreg", "node2vec_logreg", "deepwalk_logreg"):
        from gensim.models import Word2Vec

        return Word2Vec.load(str(d / f"{name}.gensim"))
    elif name == "doc2vec_logreg":
        from gensim.models.doc2vec import Doc2Vec

        return Doc2Vec.load(str(d / f"{name}.gensim"))
    else:  # sklearn / boosting / propagation dicts
        return joblib.load(d / f"{name}.joblib")
    m.load_state_dict(torch.load(d / f"{name}.pt", map_location="cpu"))
    return m.eval().to(DEVICE)


def load_extra(name: str, D: Data):
    return joblib.load(STEP1_DIR(D) / "models" / f"{name}_extra.joblib")


def stored_scores(name: str, D: Data) -> np.ndarray:
    """The score column step 1 wrote for this model (one value per account, all splits)."""
    df = pd.read_csv(STEP1_DIR(D) / "predictions" / f"{name}.csv")
    return df.sort_values("account_id")["score"].to_numpy()


def verify_reload(name: str, D: Data, new_scores, atol=1e-3, loose=False):
    """Compare freshly computed scores with the ones step 1 stored. Returns a one-row dict.

    atol is 1e-3 because float32 neural nets differ at the 1e-4 level between machines (the vanilla RNN
    does). loose=True is for doc2vec, whose inference start vector is seeded by Python's per-process
    salted hash(), so step-1 scores cannot be reproduced exactly: we require rank correlation > 0.95.
    """
    old = stored_scores(name, D)
    new = np.asarray(new_scores, dtype=float)
    assert old.shape == new.shape, f"{name}: {old.shape} vs {new.shape}"
    diff = float(np.max(np.abs(old - new)))
    rho = float(pd.Series(old).corr(pd.Series(new), method="spearman"))
    ok = (rho > 0.95) if loose else (diff <= atol)
    status = "OK" if ok else "MISMATCH"
    print(f"  reload check {name:<20} max|diff|={diff:.2e}  spearman={rho:.4f}  -> {status}")
    return {"model": name, "max_abs_diff": diff, "spearman": rho, "status": status}


# ------------------------------------------------------------------ tabular
def tabular_scorer(name: str, model, D: Data):
    """f(X) -> score, where X is an (m, 12) array in RAW feature units (same scale as accounts.csv).

    Classifiers return P(inauthentic). Anomaly detectors return an anomaly score (higher = more
    anomalous). Because the scaling is inside the function, SHAP/LIME see the raw features.
    """
    cols = D.FEATURES
    if name == "autoencoder_dense":
        mu, sd = D.scaler_tab.mean_, D.scaler_tab.scale_

        @torch.no_grad()
        def f(X):
            x = torch.as_tensor((np.asarray(X, float) - mu) / sd, dtype=torch.float32)
            return ((model(x) - x) ** 2).mean(1).numpy()

        return f
    if name in TABULAR_ANOMALY:
        # Pipeline([scaler fitted on an array, detector]) : negate score_samples as in step 1
        return lambda X: -model.score_samples(np.asarray(X, float))
    return lambda X: model.predict_proba(pd.DataFrame(np.asarray(X, float), columns=cols))[:, 1]


# ------------------------------------------------------------------ sequences
def seq_tensor(name: str, X_std: np.ndarray) -> torch.Tensor:
    """Shape standardised timelines (b, 168) for a given model: CNN wants (b,1,168), RNNs (b,168,1)."""
    t = torch.as_tensor(np.asarray(X_std), dtype=torch.float32)
    return t.unsqueeze(1) if name == "cnn1d" else t.unsqueeze(-1)


def seq_output_fn(name: str, model):
    """Differentiable function: shaped batch -> one number per series (the quantity we explain).

    Classifiers: the LOGIT (sigmoid of it is the step-1 score; the logit is the usual target for
    gradient methods). LSTM autoencoder: the reconstruction error (the step-1 anomaly score).
    """
    if name == "autoencoder_lstm":
        return lambda x: ((model(x) - x) ** 2).flatten(1).mean(1)
    return lambda x: model(x)


def seq_scores(name: str, model, X_std: np.ndarray) -> np.ndarray:
    f = seq_output_fn(name, model)
    out = []
    with torch.no_grad():
        for i in range(0, len(X_std), 1024):
            out.append(f(seq_tensor(name, X_std[i : i + 1024])).numpy())
    z = np.concatenate(out)
    return z if name == "autoencoder_lstm" else sigmoid(z)


# ------------------------------------------------------------------ text
def _make_doc2vec_deterministic():
    """gensim seeds doc2vec's inference start vector with Python's hash(), which is salted per process,
    so the same post gets a slightly different vector in every run. Explanations need a function that
    returns the same value for the same input, so we swap in a stable hash (crc32). The model weights
    are untouched; only the random start of the inference differs from step 1."""
    import zlib

    import numpy as np
    from gensim.models import doc2vec

    def stable(size, seed_string=None, hashfxn=None):
        rng = np.random.Generator(np.random.SFC64(zlib.crc32(seed_string.encode()) if seed_string else 0))
        return (rng.random(size).astype(np.float32) - 0.5) / size

    doc2vec.pseudorandom_weak_vector = stable


class TextScorer:
    """Callable: list of token lists (posts) -> P(inauthentic) per post, for one of the 3 text models.

    transformer     : sigmoid of the transformer logit for the post
    word2vec_logreg : mean word2vec vector of the post's known words -> the step-1 logistic regression
    doc2vec_logreg  : doc2vec.infer_vector(post) -> the step-1 logistic regression
    The two embedding models were trained on whole-account vectors (6 posts); applying them to a single
    post is a small distribution shift, which is noted in the notebook.
    """

    def __init__(self, name: str, model, D: Data):
        self.name, self.model = name, model
        if name == "transformer":
            ex = load_extra(name, D)
            self.vocab, self.max_len = ex["vocab"], ex["max_len"]
        else:
            self.clf = load_extra(name, D)["classifier"]
            self.dim = model.vector_size
            if name == "doc2vec_logreg":
                _make_doc2vec_deterministic()

    def encode(self, tokens):
        ids = [M.CLS] + [self.vocab.get(t, M.UNK) for t in tokens]
        return ids + [M.PAD] * (self.max_len - len(ids))

    def vectors(self, token_lists):
        if self.name == "word2vec_logreg":
            wv = self.model.wv
            return np.vstack([
                np.mean([wv[t] for t in toks if t in wv], axis=0) if any(t in wv for t in toks) else np.zeros(self.dim)
                for toks in token_lists
            ])
        return np.vstack([self.model.infer_vector(list(toks)) for toks in token_lists])

    @torch.no_grad()
    def __call__(self, token_lists):
        token_lists = [list(t) for t in token_lists]
        if self.name == "transformer":
            out = []
            for i in range(0, len(token_lists), 512):
                ids = torch.as_tensor([self.encode(t) for t in token_lists[i : i + 512]], dtype=torch.long)
                out.append(torch.sigmoid(self.model(ids)).numpy())
            return np.concatenate(out)
        return self.clf.predict_proba(self.vectors(token_lists))[:, 1]

    def account_scores(self, D: Data) -> np.ndarray:
        """Reproduce the step-1 account score (to verify the reload)."""
        if self.name == "transformer":
            p = self(D.post_tokens)
            return pd.Series(p).groupby(D.post_account).mean().reindex(np.arange(D.n)).to_numpy()
        E = self.vectors(D.account_tokens)
        if self.name == "doc2vec_logreg":  # step 1 used the stored training vectors for train accounts
            for i in np.flatnonzero(D.is_tr):
                E[i] = self.model.dv[int(i)]
        return self.clf.predict_proba(E)[:, 1]


# ------------------------------------------------------------------ graph: GNNs, embeddings, fusion
def gnn_logits(model, x: torch.Tensor, edge_index: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(x, edge_index).numpy()


def embedding_matrix(w2v, n: int) -> np.ndarray:
    dim = w2v.vector_size
    return np.vstack([w2v.wv[str(i)] if str(i) in w2v.wv else np.zeros(dim) for i in range(n)])


def fusion_inputs(D: Data, transformer):
    """x_tab (standardised profile stats) and x_txt (scaled mean [CLS] embedding of each account's posts)."""
    ex = load_extra("multimodal_fusion", D)
    ids = torch.as_tensor(
        [TextScorer("transformer", transformer, D).encode(t) for t in D.post_tokens], dtype=torch.long
    )
    with torch.no_grad():
        post_emb = torch.cat([transformer.embed(ids[i : i + 2048]) for i in range(0, len(ids), 2048)]).numpy()
    emb = np.zeros((D.n, post_emb.shape[1]))
    np.add.at(emb, D.post_account, post_emb)
    emb /= np.bincount(D.post_account, minlength=D.n)[:, None]
    x_txt = torch.as_tensor(ex["text_scaler"].transform(emb), dtype=torch.float32)
    x_tab = torch.as_tensor(D.X_tab_std, dtype=torch.float32)
    return x_tab, x_txt


# ------------------------------------------------------------------ graph: label spreading
def adjacency(n, src, dst):
    A = sparse.coo_matrix((np.ones(2 * len(src)), (np.r_[src, dst], np.r_[dst, src])), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    return A


class LabelSpreading:
    """Step-1 label spreading, in closed form: F = (1-alpha) (I - alpha S)^-1 Y0.

    Step 1 iterated 100 times; with alpha=0.8 the difference from the closed form is ~1e-10. The closed
    form makes the explanation exact: the class-1 mass at node i is a sum over seeds, seed s contributing
    K[i, s] * Y0[s, 1]. Removing seed s therefore just subtracts its term.
    """

    def __init__(self, D: Data, meta: dict, src=None, dst=None):
        self.D, self.alpha = D, meta["alpha"]
        self.seeds = np.asarray(meta["seed_accounts"])
        A = adjacency(D.n, D.src if src is None else src, D.dst if dst is None else dst)  # src/dst: optional edge subset
        deg = np.asarray(A.sum(1)).ravel()
        dinv = sparse.diags(1.0 / np.sqrt(np.maximum(deg, 1.0)))
        S = (dinv @ A @ dinv).tocsc()
        self.lu = splu((sparse.identity(D.n, format="csc") - self.alpha * S).tocsc())
        self.Y0 = np.zeros((D.n, 2))
        self.Y0[self.seeds, D.y[self.seeds]] = 1.0
        self.prior = float(D.y[self.seeds].mean())

    def kernel_row(self, i: int) -> np.ndarray:
        """K[i, :] (S is symmetric so the row equals the column)."""
        e = np.zeros(self.D.n)
        e[i] = 1.0
        return (1 - self.alpha) * self.lu.solve(e)

    def masses(self, i, removed=()):
        k = self.kernel_row(i)
        contrib1, contrib0 = k * self.Y0[:, 1], k * self.Y0[:, 0]
        keep = np.ones(self.D.n, bool)
        keep[list(removed)] = False
        return float(contrib0[keep].sum()), float(contrib1[keep].sum()), contrib1, contrib0

    def score(self, i, removed=()):
        f0, f1, _, _ = self.masses(i, removed)
        tot = f0 + f1
        return f1 / tot if tot > 1e-12 else self.prior

    def score_from_row(self, k, removed=()):
        """Score of a node given its precomputed kernel row `k`, with some seeds removed (exact, no re-solve)."""
        keep = np.ones(self.D.n, bool)
        keep[list(removed)] = False
        f1 = float((k * self.Y0[:, 1])[keep].sum())
        f0 = float((k * self.Y0[:, 0])[keep].sum())
        return f1 / (f0 + f1) if f0 + f1 > 1e-12 else self.prior

    def all_scores(self):
        F = np.zeros((self.D.n, 2))
        for c in (0, 1):
            F[:, c] = (1 - self.alpha) * self.lu.solve(self.Y0[:, c])
        tot = F.sum(1)
        return np.where(tot > 1e-12, F[:, 1] / np.maximum(tot, 1e-12), self.prior)


# ------------------------------------------------------------------ graph: belief propagation
def _sum_by_target(log_m, b_idx, n):
    """Sum the messages arriving at each node (np.bincount is much faster than np.add.at; same result up to rounding)."""
    return np.stack([np.bincount(b_idx, weights=log_m[:, c], minlength=n) for c in (0, 1)], axis=1)


def run_bp(n, src, dst, seeds, y, homophily=0.1, damping=0.5, n_iter=30, seed_strength=0.9):
    """The step-1 belief propagation (copied from the notebook), parameterised by the edge list so that
    the explanation can delete edges. Returns (score, log_messages, a_idx, b_idx, log_phi)."""
    prior = float(y[seeds].mean())
    phi = np.tile([1 - prior, prior], (n, 1))
    phi[seeds] = np.where(y[seeds, None] == 1, [1 - seed_strength, seed_strength], [seed_strength, 1 - seed_strength])
    log_phi = np.log(phi)
    psi = np.array([[0.5 + homophily, 0.5 - homophily], [0.5 - homophily, 0.5 + homophily]])
    a_idx, b_idx = np.r_[src, dst], np.r_[dst, src]
    E = len(a_idx)
    rev = (np.arange(E) + E // 2) % E
    log_m = np.zeros((E, 2))
    for _ in range(n_iter):
        incoming = _sum_by_target(log_m, b_idx, n)
        cav = log_phi[a_idx] + incoming[a_idx] - log_m[rev]
        p = np.exp(cav - cav.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        new = np.log(p @ psi)
        new -= np.log(np.exp(new).sum(1, keepdims=True))
        log_m = damping * log_m + (1 - damping) * new
    incoming = _sum_by_target(log_m, b_idx, n)
    logb = log_phi + incoming
    logb -= logb.max(1, keepdims=True)
    belief = np.exp(logb)
    return belief[:, 1] / belief.sum(1), log_m, a_idx, b_idx, log_phi


def bp_scores(D: Data, meta: dict, keep_edges=None):
    seeds = np.asarray(meta["seed_accounts"])
    s, d = (D.src, D.dst) if keep_edges is None else (D.src[keep_edges], D.dst[keep_edges])
    return run_bp(D.n, s, d, seeds, D.y, meta["homophily"], meta["damping"], meta["n_iter"])


def bp_meta(D: Data):
    """Hyperparameters that step 1 recorded in registry.json (seed_strength is not stored in the model file)."""
    reg = json.loads((STEP1_DIR(D) / "registry.json").read_text())
    hp = reg["belief_propagation"]["hyperparams"]
    return hp


def make_background(D: Data, k=100, seed=0):
    """k training accounts used as background data for SHAP / LIME."""
    rng = np.random.default_rng(seed)
    return rng.choice(np.flatnonzero(D.is_tr), size=k, replace=False)
