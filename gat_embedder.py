# src/SmartATPGPro/gat_embedder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_geometric.data import Data
from torch_geometric.utils import softmax

# 与你原来保持一致的 gate type one-hot
GATE_TYPES = ["AND", "OR", "XOR", "NOT", "NAND", "NOR", "XNOR", "BUFF", "BUF", "input_pin", "output_pin"]


def _one_hot_gate(t: str) -> np.ndarray:
    v = np.zeros(len(GATE_TYPES), dtype=np.float32)
    try:
        v[GATE_TYPES.index(t)] = 1.0
    except ValueError:
        pass
    return v


def build_pyg_from_circuit(circuit):
    """
    构图规则（保持你原来“线/输出pin为节点”的风格）：
    - 节点：circuit.gates 的 key（outputpin）
    - 边：扇入线 -> 当前线（fanin->gate输出线）
    - 为了做“方向/关系敏感”，同时加入反向边并用 edge_type 区分：
        edge_type = 0: fanin -> gate_out
        edge_type = 1: gate_out -> fanin   (reverse)
    """
    nodes = list(circuit.gates.keys())
    node_index = {nid: i for i, nid in enumerate(nodes)}

    # 逻辑层级（自底向上 BFS），与原实现一致
    level = {nid: 0 for nid in nodes}
    from collections import deque
    q = deque([g.outputpin for g in circuit.primary_input_gates])
    seen = set(q)
    while q:
        nid = q.popleft()
        for og in circuit.gates[nid].output_gates:
            lid = og.outputpin
            level[lid] = max(level[lid], level[nid] + 1)
            if lid not in seen:
                seen.add(lid)
                q.append(lid)
    max_level = max(level.values()) if level else 1

    # 边：fanin->current + reverse（用于方向关系）
    src, dst, etype = [], [], []
    for nid, gate in circuit.gates.items():
        for ig in gate.input_gates:
            s = node_index[ig.outputpin]
            d = node_index[nid]

            # fanin -> out
            src.append(s)
            dst.append(d)
            etype.append(0)

            # reverse: out -> fanin
            src.append(d)
            dst.append(s)
            etype.append(1)

    # 特征：onehot + level + fanout + SCOAP(CC0,CC1,CCb)（与你原来一致）
    X = []
    fanouts = {nid: len(circuit.gates[nid].output_gates) for nid in nodes}
    max_fanout = max(fanouts.values()) if fanouts else 1

    # 若未计算 SCOAP，调用一次（与你原来一致）
    if (
        nodes
        and circuit.gates[nodes[0]].CC0 == 0
        and circuit.gates[nodes[0]].CC1 == 0
        and nodes[0] not in [g.outputpin for g in circuit.primary_input_gates]
    ):
        try:
            circuit.calculate_SCOAP()
        except Exception:
            pass

    for nid in nodes:
        g = circuit.gates[nid]
        onehot = _one_hot_gate(g.type)
        lev = np.array([level[nid] / (max_level + 1e-6)], dtype=np.float32)
        fout = np.array([fanouts[nid] / (max_fanout + 1e-6)], dtype=np.float32)
        scoap = np.array([float(g.CC0), float(g.CC1), float(g.CCb)], dtype=np.float32)
        scoap = (scoap - scoap.mean()) / (scoap.std() + 1e-6)  # 简单标准化
        X.append(np.concatenate([onehot, lev, fout, scoap], dtype=np.float32))

    x = torch.tensor(np.vstack(X), dtype=torch.float)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_type = torch.tensor(etype, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_type=edge_type, num_nodes=len(nodes))
    data.num_relations = 2
    return data, node_index


class RGCNAttnConv(nn.Module):
    """
    RGCN + Attention（邻居级注意力）的内存友好实现：

    - relation-sensitive：每种 relation r 有 W_r
    - direction-aware：用 edge_type 表示方向/关系
    - attention：对每条入边在目标节点处做 softmax 归一化

    关键：不做 W_e = W[edge_type]（会生成 [E,H,Fin,Fout] 巨大张量导致 OOM）
    做法：按关系分组，再分块计算消息，显著降低峰值内存。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_relations: int,
        heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True,
        negative_slope: float = 0.2,
        add_root: bool = True,
        chunk_size: int = 50000,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.num_relations = int(num_relations)
        self.heads = int(heads)
        self.concat = bool(concat)
        self.negative_slope = float(negative_slope)
        self.chunk_size = int(chunk_size)

        # relation-specific projection: [R, H, Fin, Fout]
        self.W = nn.Parameter(torch.empty(self.num_relations, self.heads, self.in_channels, self.out_channels))

        # attention vectors per relation/head: [R, H, Fout]
        self.att_src = nn.Parameter(torch.empty(self.num_relations, self.heads, self.out_channels))
        self.att_dst = nn.Parameter(torch.empty(self.num_relations, self.heads, self.out_channels))

        self.dropout = nn.Dropout(float(dropout))

        self.root = None
        if add_root:
            self.root = nn.Linear(self.in_channels, self.heads * self.out_channels, bias=False)

        self.bias = nn.Parameter(torch.zeros(self.heads * self.out_channels if self.concat else self.out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        if self.root is not None:
            nn.init.xavier_uniform_(self.root.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor | None) -> torch.Tensor:
        """
        x: [N, Fin]
        edge_index: [2, E]
        edge_type: [E]
        """
        if edge_type is None:
            edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device)

        N = int(x.size(0))
        src, dst = edge_index[0], edge_index[1]

        # 先用 concat 形式累加（即 [N, H*Fout]），最后若 concat=False 再 mean heads
        out = x.new_zeros((N, self.heads * self.out_channels))

        for r in range(self.num_relations):
            mask = (edge_type == r)
            if not bool(mask.any()):
                continue

            e_idx = mask.nonzero(as_tuple=False).view(-1)
            s_all = src[e_idx]
            d_all = dst[e_idx]

            W_r = self.W[r]       # [H, Fin, Fout]
            a_s = self.att_src[r] # [H, Fout]
            a_d = self.att_dst[r] # [H, Fout]

            # -------- 1) 先算 logits（只得到 [Er, H]，避免 [Er,H,Fout] 的巨大中间量）
            x_s_all = x[s_all]  # [Er, Fin]
            x_d_all = x[d_all]  # [Er, Fin]

            # term_src = sum_k( (x_s W_r)_k * a_s_k )  -> 直接 einsum 到 [Er,H]
            term_src = torch.einsum("ei,hio,ho->eh", x_s_all, W_r, a_s)
            term_dst = torch.einsum("ei,hio,ho->eh", x_d_all, W_r, a_d)

            logits = term_src + term_dst
            logits = F.leaky_relu(logits, negative_slope=self.negative_slope)  # [Er,H]

            alpha = softmax(logits, d_all, num_nodes=N)  # [Er,H]
            alpha = self.dropout(alpha)

            # -------- 2) 再分块算消息并聚合（只在块内生成 [B,H,Fout]）
            Er = int(s_all.size(0))
            cs = self.chunk_size if self.chunk_size > 0 else Er

            for start in range(0, Er, cs):
                end = min(start + cs, Er)
                s = s_all[start:end]
                d = d_all[start:end]
                a = alpha[start:end]  # [B,H]

                x_s = x[s]  # [B,Fin]
                h_s = torch.einsum("ei,hio->eho", x_s, W_r)  # [B,H,Fout]

                m = (h_s * a.unsqueeze(-1)).reshape(h_s.size(0), -1)  # [B, H*Fout]
                out.index_add_(0, d, m)

        # root/skip
        if self.root is not None:
            out = out + self.root(x)

        if self.concat:
            out = out + self.bias
            return out  # [N, H*Fout]
        else:
            out = out.view(N, self.heads, self.out_channels).mean(dim=1)  # [N, Fout]
            out = out + self.bias
            return out


class RGCNAttnEmbedder(nn.Module):
    """两层 RGCN-Attn，接口保持与原 GATEmbedder 一致：输出 [N, out_dim]"""

    def __init__(self, in_dim, hidden=128, out_dim=64, heads=8, dropout=0.1, num_relations: int = 2):
        super().__init__()
        self.num_relations = int(num_relations)

        # 第一层：多头 concat -> [N, heads*hidden]
        self.conv1 = RGCNAttnConv(
            in_channels=int(in_dim),
            out_channels=int(hidden),
            num_relations=self.num_relations,
            heads=int(heads),
            dropout=float(dropout),
            concat=True,
            add_root=True,
            chunk_size=50000,
        )

        # 第二层：heads=1, concat=False -> [N, out_dim]
        self.conv2 = RGCNAttnConv(
            in_channels=int(hidden) * int(heads),
            out_channels=int(out_dim),
            num_relations=self.num_relations,
            heads=1,
            dropout=float(dropout),
            concat=False,
            add_root=True,
            chunk_size=50000,
        )

        self.act = nn.ELU()

    def forward(self, data: Data):
        x = self.conv1(data.x, data.edge_index, getattr(data, "edge_type", None))
        x = self.act(x)
        x = self.conv2(x, data.edge_index, getattr(data, "edge_type", None))
        return x  # [N, out_dim]


# 为了不改其他文件：仍然导出同名 GATEmbedder（但内部已是 RGCN-Attn）
GATEmbedder = RGCNAttnEmbedder


def compute_embeddings(circuit, hidden=128, out_dim=64, heads=8, device="gpu"):
    """
    与原接口保持一致：
    emb: numpy [N, out_dim]
    node_map: dict[node_id -> index]
    """
    # 兼容你原来的 device="gpu" 写法
    if device == "gpu":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    data, node_map = build_pyg_from_circuit(circuit)

    num_rel = int(getattr(data, "num_relations", 2))
    model = RGCNAttnEmbedder(
        in_dim=data.num_node_features,
        hidden=hidden,
        out_dim=out_dim,
        heads=heads,
        dropout=0.1,
        num_relations=num_rel,
    ).to(device)

    model.eval()
    with torch.no_grad():
        emb = model(data.to(device)).cpu().numpy()

    return emb, node_map
