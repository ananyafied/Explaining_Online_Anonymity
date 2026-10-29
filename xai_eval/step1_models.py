"""Model definitions copied from notebooks/01_fit_models.ipynb.

Step 1 saved neural-network *weights only* (state_dict), so these classes must stay identical to
the ones in that notebook, otherwise the weights will not load (or, worse, load into the wrong
architecture). `loaders.verify_reload` checks that the reloaded models reproduce the stored
step-1 predictions, which catches any drift.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, CLS, UNK = 0, 1, 2


class SeqCNN(nn.Module):
    """1D CNN over the posting timeline. `features` is the last conv block (the Grad-CAM target layer)."""

    def __init__(self, channels=(16, 32), k=5, dropout=0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, channels[0], k, padding=k // 2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(channels[0], channels[1], k, padding=k // 2),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(channels[1], 1))

    def forward(self, x):  # x: (batch, 1, 168)
        return self.head(self.pool(self.features(x)).squeeze(-1)).squeeze(-1)


class SeqRNN(nn.Module):
    """Vanilla RNN or LSTM; the last time step's output goes to a linear head."""

    def __init__(self, kind="lstm", hidden=32):
        super().__init__()
        cls = {"rnn": nn.RNN, "lstm": nn.LSTM}[kind]
        self.rnn = cls(1, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):  # x: (batch, 168, 1)
        out, _ = self.rnn(x)
        return self.head(out[:, -1]).squeeze(-1)


class PostTransformer(nn.Module):
    """Small transformer encoder. A [CLS] token is prepended and its output is classified."""

    def __init__(self, vocab_size, max_len, d_model=64, nhead=4, n_layers=2, d_ff=128, dropout=0.1):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, d_ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, 1)

    def embed(self, ids):  # (batch, L) -> (batch, d_model): the [CLS] representation
        pos = torch.arange(ids.size(1), device=ids.device).unsqueeze(0)
        h = self.encoder(self.tok(ids) + self.pos(pos), src_key_padding_mask=(ids == PAD))
        return h[:, 0]

    def forward(self, ids):
        return self.head(self.embed(ids)).squeeze(-1)


class DenseAE(nn.Module):
    def __init__(self, d_in, bottleneck=4):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, 8), nn.ReLU(), nn.Linear(8, bottleneck))
        self.dec = nn.Sequential(nn.Linear(bottleneck, 8), nn.ReLU(), nn.Linear(8, d_in))

    def forward(self, x):
        return self.dec(self.enc(x))


class LSTMAutoencoder(nn.Module):
    """Encode the whole timeline into the final hidden state, then decode it back step by step."""

    def __init__(self, hidden=32):
        super().__init__()
        self.enc = nn.LSTM(1, hidden, batch_first=True)
        self.dec = nn.LSTM(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):  # x: (batch, 168, 1)
        _, (h, _) = self.enc(x)
        z = h[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        d, _ = self.dec(z)
        return self.out(d)


def make_gnn(kind, d_in, hidden=64, dropout=0.3):
    """Two-layer node classifier. kind in {gcn, gat, sage, rgcn}."""
    from torch_geometric.nn import GATConv, GCNConv, RGCNConv, SAGEConv

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.kind, self.dropout = kind, dropout
            if kind == "gcn":
                self.c1, self.c2 = GCNConv(d_in, hidden), GCNConv(hidden, hidden)
            elif kind == "gat":
                self.c1 = GATConv(d_in, hidden // 4, heads=4)
                self.c2 = GATConv(hidden, hidden // 4, heads=4)
            elif kind == "sage":
                self.c1, self.c2 = SAGEConv(d_in, hidden), SAGEConv(hidden, hidden)
            elif kind == "rgcn":  # one relation type only
                self.c1 = RGCNConv(d_in, hidden, num_relations=1)
                self.c2 = RGCNConv(hidden, hidden, num_relations=1)
            else:
                raise ValueError(kind)
            self.out = nn.Linear(hidden, 1)

        def conv(self, layer, h, edge_index):
            if self.kind == "rgcn":
                return layer(h, edge_index, torch.zeros(edge_index.size(1), dtype=torch.long, device=h.device))
            return layer(h, edge_index)

        def forward(self, x, edge_index):
            h = F.dropout(x, self.dropout, self.training)
            h = F.relu(self.conv(self.c1, h, edge_index))
            h = F.dropout(h, self.dropout, self.training)
            h = F.relu(self.conv(self.c2, h, edge_index))
            return self.out(h).squeeze(-1)

    return Net()


def make_fusion(d_tab, d_txt, hidden=64, dropout=0.3):
    from torch_geometric.nn import GCNConv

    class Fusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = dropout
            self.tab_proj = nn.Linear(d_tab, hidden // 2)
            self.txt_proj = nn.Linear(d_txt, hidden // 2)
            self.c1, self.c2 = GCNConv(hidden, hidden), GCNConv(hidden, hidden)
            self.out = nn.Linear(2 * hidden, 1)

        def forward(self, x_tab, x_txt, edge_index):
            h0 = F.relu(torch.cat([self.tab_proj(x_tab), self.txt_proj(x_txt)], dim=1))
            h = F.dropout(h0, self.dropout, self.training)
            h = F.relu(self.c1(h, edge_index))
            h = F.dropout(h, self.dropout, self.training)
            h = F.relu(self.c2(h, edge_index))
            return self.out(torch.cat([h0, h], dim=1)).squeeze(-1)

    return Fusion()
