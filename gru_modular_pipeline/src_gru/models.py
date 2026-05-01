"""PyTorch datasets and models for the modular GRU/LSTM M5 pipeline."""

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class M5WindowDataset(Dataset):
    """Dataset for direct 28-day forecasting windows."""

    def __init__(self, X, y, X_cat, X_num, X_future_cal, X_future_event, X_price):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.X_future_cal = torch.tensor(X_future_cal, dtype=torch.float32)
        self.X_future_event = torch.tensor(X_future_event, dtype=torch.long)
        self.X_price = torch.tensor(X_price, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            self.X[idx],
            self.y[idx],
            self.X_cat[idx],
            self.X_num[idx],
            self.X_future_cal[idx],
            self.X_future_event[idx],
            self.X_price[idx],
        )


class DirectRNNForecast(nn.Module):
    """GRU/LSTM direct-output model with static embeddings and engineered features."""

    def __init__(
        self,
        cat_cardinalities,
        event_type_cardinality,
        num_numeric_features,
        num_future_cal_features,
        num_price_features,
        cfg,
    ):
        super().__init__()
        model_cfg = cfg["model"]
        forecast_cfg = cfg["forecast"]

        self.horizon = int(forecast_cfg["horizon"])
        self.rnn_type = model_cfg.get("rnn_type", "gru").lower()
        hidden_size = int(model_cfg["hidden_size"])
        num_layers = int(model_cfg["num_layers"])
        dropout = float(model_cfg["dropout"])

        rnn_cls = nn.LSTM if self.rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.cat_embs = nn.ModuleList([
            nn.Embedding(int(cardinality), min(50, max(2, (int(cardinality) + 1) // 2)))
            for cardinality in cat_cardinalities
        ])
        cat_emb_dim = sum(emb.embedding_dim for emb in self.cat_embs)

        event_dim = int(model_cfg.get("event_embedding_dim", 4))
        self.event_emb = nn.Embedding(int(event_type_cardinality), event_dim)

        total_dim = (
            hidden_size
            + cat_emb_dim
            + int(num_numeric_features)
            + int(num_future_cal_features) * self.horizon
            + event_dim * self.horizon
            + int(num_price_features)
        )

        layers = [
            nn.Linear(total_dim, int(model_cfg.get("head_hidden_size", 128))),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(model_cfg.get("head_hidden_size", 128)), self.horizon),
        ]
        if model_cfg.get("use_softplus_output", True):
            layers.append(nn.Softplus())
        self.head = nn.Sequential(*layers)

    def forward(self, x, x_cat, x_num, x_future_cal, x_future_event, x_price):
        x = x.unsqueeze(-1)
        if self.rnn_type == "lstm":
            _, (h, _) = self.rnn(x)
        else:
            _, h = self.rnn(x)
        h = h[-1]

        cat_emb = torch.cat(
            [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embs)], dim=1
        )
        future_cal_flat = x_future_cal.reshape(x_future_cal.size(0), -1)
        event_flat = self.event_emb(x_future_event).reshape(x_future_event.size(0), -1)

        parts = [h, cat_emb, x_num, future_cal_flat, event_flat]
        if x_price.numel() > 0:
            parts.append(x_price)
        z = torch.cat(parts, dim=1)
        return self.head(z)


def build_model(cat_sizes, event_type_cardinality, feature_shapes, cfg, device):
    """Factory to build and move the model to device."""
    model = DirectRNNForecast(
        cat_cardinalities=cat_sizes,
        event_type_cardinality=event_type_cardinality,
        num_numeric_features=feature_shapes["num_numeric_features"],
        num_future_cal_features=feature_shapes["num_future_cal_features"],
        num_price_features=feature_shapes["num_price_features"],
        cfg=cfg,
    )
    return model.to(device)
