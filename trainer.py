import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import pandas as pd
from typing import List

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.nn import SAGEConv


# -----------------------------
# connectable: ΔavgWire 阈值查表
# -----------------------------
def connectable(c1, c2, avgWire_diff_max: List[List[float]]):
    L1, L2 = int(c1["superlayer"]), int(c2["superlayer"])
    dL = abs(L2 - L1)
    if dL == 0 or dL > len(avgWire_diff_max):
        return False

    lower = min(L1, L2)
    thresholds = avgWire_diff_max[dL - 1]
    if lower - 1 >= len(thresholds):
        return False
    max_diff = thresholds[lower - 1]
    return abs(c1["avgWire"] - c2["avgWire"]) <= max_diff


# -----------------------------
# 解析 trkIds
# -----------------------------
def parse_trkids_str(s):
    if pd.isna(s):
        return []
    ss = str(s).strip()
    if ss == "" or ss == "-1":
        return []
    return [int(x) for x in ss.split(";") if x.strip() != "" and int(x) != -1]


# -----------------------------
# 构建单事件图
# -----------------------------
def build_graph_from_event(
    evt: pd.DataFrame,
    num_tracks: int = 2,
    avgWire_diff_max: List[List[float]] = None,
    bidirectional: bool = True
):
    if avgWire_diff_max is None:
        avgWire_diff_max = [
            [12.0, 20.0, 12.0, 38.0, 14.0],
            [14.0, 18.0, 40.0, 40.0],
            [24.0, 48.0, 48.0],
            [55.0, 55.0],
            [56.0],
        ]

    n = len(evt)
    if n == 0:
        return None

    avgWire = evt["avgWire"].values.astype(np.float32)
    superlayer = evt["superlayer"].values.astype(np.int32)

    wmin, wmax = float(avgWire.min()), float(avgWire.max())
    wrange = (wmax - wmin) if (wmax - wmin) > 1e-8 else 1.0
    avgWire_norm = (avgWire - wmin) / wrange
    superlayer_norm = superlayer / 6.0

    trk_lists = [parse_trkids_str(t) for t in evt["trkIds"].values]
    node_label = np.zeros((n, num_tracks), dtype=np.float32)
    for i, lst in enumerate(trk_lists):
        for t in lst:
            if 1 <= t <= num_tracks:
                node_label[i, t - 1] = 1.0

    degrees = np.zeros(n, dtype=np.float32)
    for i in range(n):
        deg = 0
        for j in range(n):
            if i == j:
                continue
            if connectable(evt.iloc[i], evt.iloc[j], avgWire_diff_max):
                deg += 1
        degrees[i] = deg
    max_deg = degrees.max() if degrees.max() > 0 else 1.0
    deg_norm = degrees / max_deg

    x = np.stack([avgWire_norm, superlayer_norm, deg_norm], axis=1).astype(np.float32)
    x = torch.tensor(x, dtype=torch.float)

    edges = []
    edge_attrs = []
    for i in range(n):
        for j in range(i + 1, n):
            if connectable(evt.iloc[i], evt.iloc[j], avgWire_diff_max):
                edges.append([i, j])
                edge_attrs.append([
                    float(evt.iloc[j]["superlayer"] - evt.iloc[i]["superlayer"]),
                    float((evt.iloc[j]["avgWire"] - evt.iloc[i]["avgWire"]) / wrange)
                ])

                edges.append([j, i])
                edge_attrs.append([
                    float(evt.iloc[i]["superlayer"] - evt.iloc[j]["superlayer"]),
                    float((evt.iloc[i]["avgWire"] - evt.iloc[j]["avgWire"]) / wrange)
                ])

    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 2), dtype=torch.float)
    else:
        edge_index = torch.tensor(np.array(edges).T, dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_attrs), dtype=torch.float)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor(node_label, dtype=torch.float),
        event_id=torch.tensor([int(evt["eventIdx"].iloc[0])], dtype=torch.long)
    )
    return data


# -----------------------------
# InMemoryDataset
# -----------------------------
class EventGraphDataset(InMemoryDataset):
    def __init__(self, df: pd.DataFrame, num_tracks: int = 2, avgWire_diff_max: List[List[float]] = None, transform=None):
        super().__init__(None, transform)
        self.num_tracks = num_tracks
        self.avgWire_diff_max = avgWire_diff_max
        self.data_list: List[Data] = []
        for _, evt in df.groupby("eventIdx"):
            evt = evt.reset_index(drop=True)
            data = build_graph_from_event(evt, num_tracks=self.num_tracks, avgWire_diff_max=self.avgWire_diff_max)
            if data is not None:
                self.data_list.append(data)
        self.data, self.slices = self.collate(self.data_list)

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


# -----------------------------
# Edge-attr enhanced ClusterGNN as LightningModule
# -----------------------------
class ClusterGNN(pl.LightningModule):
    """
    Edge-attr enhanced GraphSAGE for node classification, supports batch_size > 1 safely.
    """
    def __init__(
        self,
        in_channels=3,
        hidden_channels=64,
        num_layers=2,
        num_tracks=2,
        edge_feat_dim=2,
        lr=1e-3,
        dropout=0.1
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_tracks = num_tracks
        self.lr = lr
        self.dropout = dropout
        self.edge_feat_dim = edge_feat_dim

        # GraphSAGE layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            self.convs.append(SAGEConv(in_ch, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Edge MLP
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels + edge_feat_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Node classifier
        self.classifier = nn.Linear(hidden_channels, num_tracks)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, batch: Tensor = None):
        # Node embeddings via GraphSAGE
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Edge-attribute fusion
        if edge_index.numel() > 0 and edge_attr.numel() > 0:
            src, dst = edge_index[0], edge_index[1]
            edge_feat = torch.cat([x[src], x[dst], edge_attr], dim=1)
            edge_messages = self.edge_mlp(edge_feat)

            # 如果有 batch 张量，按图分开聚合
            if batch is None:
                node_messages = torch.zeros_like(x)
                node_messages = node_messages.index_add(0, src, edge_messages)
                node_messages = node_messages.index_add(0, dst, edge_messages)
            else:
                node_messages = torch.zeros_like(x)
                # 遍历每个图单独聚合
                for graph_id in batch.unique():
                    mask = (batch == graph_id)
                    node_idx = mask.nonzero(as_tuple=True)[0]
                    # 选出本图的节点和边
                    node_set = node_idx
                    # edge mask
                    edge_mask = mask[src] & mask[dst]
                    if edge_mask.sum() == 0:
                        continue
                    e_src = src[edge_mask]
                    e_dst = dst[edge_mask]
                    e_msg = edge_messages[edge_mask]

                    node_messages.index_add_(0, e_src, e_msg)
                    node_messages.index_add_(0, e_dst, e_msg)

            x = x + node_messages

        logits = self.classifier(x)
        return logits

    def compute_loss(self, logits, labels):
        pos_count = labels.sum(dim=0)
        neg_count = labels.size(0) - pos_count
        pos_weight = (neg_count / (pos_count + 1e-6)).clamp(min=1.0).to(logits.device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        return loss_fn(logits, labels)

    # -------------------------
    # Lightning interface
    # -------------------------
    def training_step(self, batch, batch_idx):
        logits = self(batch.x, batch.edge_index, batch.edge_attr, batch=batch.batch)
        loss = self.compute_loss(logits, batch.y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch.x, batch.edge_index, batch.edge_attr, batch=batch.batch)
        loss = self.compute_loss(logits, batch.y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# -----------------------------
# ClusterGNNWrapper
# -----------------------------
class ClusterGNNWrapper(nn.Module):
    """
    TorchScript-safe, GPU-friendly ClusterGNN wrapper for single-graph inference.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        # GraphSAGE convs
        self.convs = model.convs
        # BatchNorm -> Identity
        self.bns = nn.ModuleList([nn.Identity() for _ in model.bns])
        # Edge MLP
        self.edge_mlp = model.edge_mlp
        # Node classifier
        self.classifier = model.classifier
        # Dropout
        self.dropout = getattr(model, "dropout", 0.1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        # 如果节点为空，返回空 logits
        if x.numel() == 0:
            return torch.zeros((0, self.classifier.out_features), dtype=x.dtype, device=x.device)

        h = x
        # GraphSAGE 层
        for conv, bn in zip(self.convs, self.bns):
            if edge_index.numel() > 0:
                h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        # Edge message passing
        if edge_index.numel() == 0 or edge_attr.numel() == 0:
            return self.classifier(h)

        src, dst = edge_index[0], edge_index[1]
        edge_feat = torch.cat([h[src], h[dst], edge_attr], dim=1)
        edge_messages = self.edge_mlp(edge_feat)

        # 聚合消息到节点
        node_messages = torch.zeros_like(h)
        node_messages.index_add_(0, src, edge_messages)
        node_messages.index_add_(0, dst, edge_messages)

        h = h + node_messages
        logits = self.classifier(h)
        return logits

# -----------------------------
# Loss tracker callback
# -----------------------------
class LossTracker(Callback):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        if "train_loss" in trainer.callback_metrics:
            self.train_losses.append(trainer.callback_metrics["train_loss"].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        if "val_loss" in trainer.callback_metrics:
            self.val_losses.append(trainer.callback_metrics["val_loss"].item())
