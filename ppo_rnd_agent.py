# src/SmartATPGPro/ppo_rnd_agent.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import clip_grad_norm_


class MLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1])]
            if i < len(dims) - 2:
                layers += [nn.ReLU()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ActorCriticFixed(nn.Module):
    def __init__(self, node_emb_dim, state_dim, max_actions, hidden=128):
        super().__init__()
        in_dim = node_emb_dim + state_dim
        self.actor = MLP([in_dim, hidden, hidden, max_actions])
        self.critic = MLP([in_dim, hidden, 1])

    def forward(self, z, s, mask):
        x = torch.cat([z, s], dim=-1)
        logits = self.actor(x)
        logits = logits + (mask + 1e-8).log()
        value = self.critic(x).squeeze(-1)
        return logits, value


class ActorCriticDynamic(nn.Module):
    def __init__(self, node_emb_dim, state_dim, hidden=128):
        super().__init__()
        self.scorer = MLP([node_emb_dim * 2 + state_dim, hidden, hidden, 1])
        self.critic = MLP([node_emb_dim + state_dim, hidden, 1])

    def score(self, z_obj, z_cands, s):
        B, K, D = z_cands.shape
        z_obj_exp = z_obj.unsqueeze(1).expand(-1, K, -1)
        s_exp = s.unsqueeze(1).expand(-1, K, -1)
        x = torch.cat([z_obj_exp, z_cands, s_exp], dim=-1)
        logits = self.scorer(x).squeeze(-1)
        v = self.critic(torch.cat([z_obj, s], dim=-1)).squeeze(-1)
        return logits, v


class RNDModule(nn.Module):
    def __init__(self, input_dim, hidden=128, lr=1e-4):
        super().__init__()
        self.target = MLP([input_dim, hidden, hidden, hidden])
        self.predictor = MLP([input_dim, hidden, hidden, hidden])
        for p in self.target.parameters():
            p.requires_grad = False
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)

    def forward(self, x):
        with torch.no_grad():
            target_feat = self.target(x)
        pred_feat = self.predictor(x)
        loss = F.mse_loss(pred_feat, target_feat)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        err = (pred_feat - target_feat).pow(2).mean(dim=-1)
        return err.detach()


class PPOAgent:
    def __init__(
        self,
        node_embeddings,
        node_map,
        state_dim,
        max_actions=None,
        arch="dynamic",
        lr_actor=3e-4,
        lr_critic=1e-3,
        device="cpu",
        use_rnd=True,
        rnd_scale=0.05,
    ):
        self.device = device

        if node_embeddings is None:
            node_embeddings = np.zeros((1, 64), dtype=np.float32)
        if node_map is None:
            node_map = {}

        self.E = torch.tensor(node_embeddings, dtype=torch.float, device=device)
        self.node_map = node_map
        self.state_dim = int(state_dim)
        self.node_emb_dim = int(self.E.shape[1]) if self.E.ndim == 2 else 0
        self.arch = arch
        self.max_actions = int(max_actions) if max_actions is not None else None

        self.use_rnd = use_rnd
        self.rnd_scale = float(rnd_scale)

        self.gat_meta = {}

        if self.arch == "fixed":
            assert self.max_actions is not None, "fixed 架构需要 max_actions"
            self.ac_fixed = ActorCriticFixed(self.node_emb_dim, self.state_dim, self.max_actions).to(device)
            self.optim_actor = torch.optim.Adam(self.ac_fixed.actor.parameters(), lr=lr_actor)
            self.optim_critic = torch.optim.Adam(self.ac_fixed.critic.parameters(), lr=lr_critic)
        else:
            self.ac_dyn = ActorCriticDynamic(self.node_emb_dim, self.state_dim).to(device)
            self.optim_actor = torch.optim.Adam(self.ac_dyn.scorer.parameters(), lr=lr_actor)
            self.optim_critic = torch.optim.Adam(self.ac_dyn.critic.parameters(), lr=lr_critic)

        self.rnd = None
        if self.use_rnd:
            rnd_in = self.node_emb_dim + self.state_dim
            if rnd_in <= 0:
                rnd_in = self.state_dim
            self.rnd = RNDModule(rnd_in, hidden=128, lr=1e-4).to(self.device)

    def set_graph_embeddings(self, node_embeddings, node_map):
        E_new = torch.tensor(node_embeddings, dtype=torch.float, device=self.device)
        if E_new.ndim != 2:
            raise ValueError("node_embeddings must be 2-D [N, D]")
        if int(E_new.shape[1]) != int(self.node_emb_dim):
            raise ValueError(
                f"Embedding dim mismatch: got {int(E_new.shape[1])}, expected {int(self.node_emb_dim)}"
            )
        self.E = E_new
        self.node_map = node_map if node_map is not None else {}

    def _idx(self, node_id):
        return self.node_map.get(node_id, 0)

    @staticmethod
    def _pad_or_trunc(arr, target_len):
        arr = np.asarray(arr)
        cur = len(arr)
        if cur == target_len:
            return arr
        if cur < target_len:
            pad = np.zeros(target_len - cur, dtype=arr.dtype)
            return np.concatenate([arr, pad])
        return arr[:target_len]

    @staticmethod
    def _normalize_gate_type(t: str) -> str:
        if t is None:
            return "BUF"
        t = str(t)
        if t == "BUFF":
            return "BUF"
        return t

    def _state_to_vec(self, state_dict) -> np.ndarray:
        """
        新版 state 编码（与 PODEM._build_state_for_policy 对齐）：
        [gate_type_onehot,
         level_norm, fanout_norm,
         log1p(CC0,CC1,CCb),
         log1p(fault_bt_sa0,fault_bt_sa1),
         phase, obj_val, depth_norm, cand_count_norm]
        """
        if state_dict is None:
            return np.zeros((self.state_dim,), dtype=np.float32)

        if "state_vec" in state_dict and state_dict["state_vec"] is not None:
            s_np = np.asarray(state_dict["state_vec"], dtype=np.float32).reshape(-1)
            return self._pad_or_trunc(s_np, self.state_dim).astype(np.float32)

        try:
            from SmartATPGPro.gat_embedder import GATE_TYPES
        except Exception:
            GATE_TYPES = ["AND", "OR", "XOR", "NOT", "NAND", "NOR", "XNOR", "BUFF", "BUF", "input_pin", "output_pin"]

        gate_type = self._normalize_gate_type(state_dict.get("gate_type", "BUF"))
        onehot = np.zeros(len(GATE_TYPES), dtype=np.float32)
        if gate_type in GATE_TYPES:
            onehot[GATE_TYPES.index(gate_type)] = 1.0

        eps = 1e-6

        level_norm = state_dict.get("level_norm", None)
        if level_norm is None:
            lvl = float(state_dict.get("level", 0.0))
            max_level = float(state_dict.get("max_level", 1.0))
            level_norm = lvl / (max_level + eps)

        fanout_norm = state_dict.get("fanout_norm", None)
        if fanout_norm is None:
            fo = float(state_dict.get("fanout", 0.0))
            max_fanout = float(state_dict.get("max_fanout", 1.0))
            fanout_norm = fo / (max_fanout + eps)

        level_norm = np.array([float(level_norm)], dtype=np.float32)
        fanout_norm = np.array([float(fanout_norm)], dtype=np.float32)

        cc0 = float(state_dict.get("CC0", 0.0))
        cc1 = float(state_dict.get("CC1", 0.0))
        ccb = float(state_dict.get("CCb", 0.0))
        scoap = np.log1p(np.maximum(np.array([cc0, cc1, ccb], dtype=np.float32), 0.0))

        bt0 = float(state_dict.get("fault_bt_sa0", 0.0))
        bt1 = float(state_dict.get("fault_bt_sa1", 0.0))
        bt = np.log1p(np.maximum(np.array([bt0, bt1], dtype=np.float32), 0.0))

        # 新增 4 个字段
        phase = np.array([float(state_dict.get("phase", 0.0))], dtype=np.float32)
        obj_val = np.array([float(state_dict.get("obj_val", 0.5))], dtype=np.float32)
        depth_norm2 = np.array([float(state_dict.get("depth_norm", 0.0))], dtype=np.float32)
        cand_norm = np.array([float(state_dict.get("cand_count_norm", 0.0))], dtype=np.float32)

        vec = np.concatenate([onehot, level_norm, fanout_norm, scoap, bt, phase, obj_val, depth_norm2, cand_norm]).astype(
            np.float32
        )
        return self._pad_or_trunc(vec, self.state_dim).astype(np.float32)

    def encode_state_vec(self, state_dict) -> np.ndarray:
        return self._state_to_vec(state_dict).astype(np.float32, copy=True)

    def act(self, state_dict, deterministic=True):
        if self.arch == "dynamic" and "cand_ids" in state_dict:
            with torch.no_grad():
                node_id = state_dict.get("node_id", state_dict.get("node", None))

                if "gat_vec" in state_dict and state_dict["gat_vec"] is not None and self.node_emb_dim > 0:
                    gv = np.asarray(state_dict["gat_vec"], dtype=np.float32).reshape(-1)
                    gv = self._pad_or_trunc(gv, self.node_emb_dim)
                    z_obj = torch.tensor(gv, dtype=torch.float, device=self.device).unsqueeze(0)
                else:
                    idx_obj = self._idx(node_id)
                    idx_obj = max(0, min(int(idx_obj), int(self.E.shape[0]) - 1))
                    z_obj = self.E[idx_obj].unsqueeze(0)

                s_np = self._state_to_vec(state_dict)
                s = torch.tensor(s_np, dtype=torch.float, device=self.device).unsqueeze(0)

                cand_ids = list(state_dict["cand_ids"])
                if len(cand_ids) == 0:
                    return 0, 0.0, 0.0
                cand_idx = [max(0, min(int(self._idx(cid)), int(self.E.shape[0]) - 1)) for cid in cand_ids]
                z_cands = self.E[cand_idx].unsqueeze(0)

                logits, value = self.ac_dyn.score(z_obj, z_cands, s)
                if "mask" in state_dict and state_dict["mask"] is not None:
                    mask = torch.tensor(np.asarray(state_dict["mask"], dtype=np.float32), device=self.device).view(1, -1)
                    logits = logits + (mask + 1e-8).log()

                if deterministic:
                    a = int(torch.argmax(logits, dim=-1).item())
                    logp = 0.0
                else:
                    probs = F.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    a = int(dist.sample().item())
                    logp = float(dist.log_prob(torch.tensor(a, device=self.device)).item())
                v = float(value.item())
                return a, logp, v

        self.ac_fixed.eval()
        with torch.no_grad():
            node_id = state_dict.get("node_id", state_dict.get("node", None))
            idx = self._idx(node_id)

            s_np = self._state_to_vec(state_dict)
            m_np = np.asarray(state_dict["mask"], dtype=np.float32).reshape(-1)
            s_np = self._pad_or_trunc(s_np, self.state_dim)
            m_np = self._pad_or_trunc(m_np, self.max_actions)

            if "gat_vec" in state_dict and state_dict["gat_vec"] is not None and self.node_emb_dim > 0:
                gv = np.asarray(state_dict["gat_vec"], dtype=np.float32).reshape(-1)
                gv = self._pad_or_trunc(gv, self.node_emb_dim)
                z = torch.tensor(gv, dtype=torch.float, device=self.device)
            elif self.E.ndim == 2 and self.E.shape[0] > 0:
                idx = max(0, min(int(idx), self.E.shape[0] - 1))
                z = self.E[idx]
            else:
                z = torch.zeros((self.node_emb_dim,), dtype=torch.float, device=self.device)

            s = torch.tensor(s_np, dtype=torch.float, device=self.device).unsqueeze(0)
            m = torch.tensor(m_np, dtype=torch.float, device=self.device).unsqueeze(0)
            z = z.unsqueeze(0)

            logits, value = self.ac_fixed(z, s, m)
            if deterministic:
                a = int(torch.argmax(logits, dim=-1).item())
                logp = 0.0
            else:
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                a = int(dist.sample().item())
                logp = float(dist.log_prob(torch.tensor(a, device=self.device)).item())
            v = float(value.item())
            return a, logp, v

    def score_candidates(self, state_dict):
        if self.arch != "dynamic":
            raise RuntimeError("score_candidates only supports arch='dynamic'")
        if "cand_ids" not in state_dict:
            raise ValueError("state_dict must contain 'cand_ids' for dynamic scoring")

        with torch.no_grad():
            node_id = state_dict.get("node_id", state_dict.get("node", None))
            if "gat_vec" in state_dict and state_dict["gat_vec"] is not None and self.node_emb_dim > 0:
                gv = np.asarray(state_dict["gat_vec"], dtype=np.float32).reshape(-1)
                gv = self._pad_or_trunc(gv, self.node_emb_dim)
                z_obj = torch.tensor(gv, dtype=torch.float, device=self.device).unsqueeze(0)
            else:
                idx_obj = self._idx(node_id)
                idx_obj = max(0, min(int(idx_obj), int(self.E.shape[0]) - 1))
                z_obj = self.E[idx_obj].unsqueeze(0)

            s_np = self._state_to_vec(state_dict)
            s = torch.tensor(s_np, dtype=torch.float, device=self.device).unsqueeze(0)

            cand_ids = list(state_dict["cand_ids"])
            if len(cand_ids) == 0:
                return np.zeros((0,), dtype=np.float32), 0.0, np.zeros((0,), dtype=np.float32), 0.0, 0.0
            cand_idx = [max(0, min(int(self._idx(cid)), int(self.E.shape[0]) - 1)) for cid in cand_ids]
            z_cands = self.E[cand_idx].unsqueeze(0)

            logits, value = self.ac_dyn.score(z_obj, z_cands, s)
            logits = logits.squeeze(0)

            if "mask" in state_dict and state_dict["mask"] is not None:
                mask = torch.tensor(np.asarray(state_dict["mask"], dtype=np.float32), device=self.device).view(-1)
                logits = logits + (mask + 1e-8).log()

            probs = F.softmax(logits, dim=-1)
            ent = float(torch.distributions.Categorical(probs).entropy().mean().item())
            mx = float(torch.max(probs).item())
            return (
                logits.detach().cpu().numpy().astype(np.float32),
                float(value.item()),
                probs.detach().cpu().numpy().astype(np.float32),
                ent,
                mx,
            )

    def intrinsic_reward(self, state_dict):
        if self.rnd is None:
            return 0.0

        node_id = state_dict.get("node_id", state_dict.get("node", None))
        if "gat_vec" in state_dict and state_dict["gat_vec"] is not None and self.node_emb_dim > 0:
            gv = np.asarray(state_dict["gat_vec"], dtype=np.float32).reshape(-1)
            gv = self._pad_or_trunc(gv, self.node_emb_dim)
            z_obj = torch.tensor(gv, dtype=torch.float, device=self.device).detach()
        else:
            idx_obj = self._idx(node_id)
            idx_obj = max(0, min(int(idx_obj), int(self.E.shape[0]) - 1))
            z_obj = self.E[idx_obj].detach()

        s_np = self._state_to_vec(state_dict)
        s = torch.tensor(s_np, dtype=torch.float, device=self.device)

        if self.node_emb_dim > 0:
            x = torch.cat([z_obj, s], dim=-1).unsqueeze(0)
        else:
            x = s.unsqueeze(0)

        with torch.enable_grad():
            novelty = self.rnd(x)[0]
        return float(self.rnd_scale * novelty.item())

    def save(self, path, save_graph: bool = True):
        meta = {
            "arch": self.arch,
            "state_dim": self.state_dim,
            "node_emb_dim": self.node_emb_dim,
            "use_rnd": self.use_rnd,
            "rnd_scale": self.rnd_scale,
            "gat_meta": getattr(self, "gat_meta", {}),
        }
        payload = {"meta": meta}

        if save_graph:
            payload["E"] = self.E.detach().cpu().numpy()
            payload["node_map"] = self.node_map

        if self.arch == "fixed":
            meta["max_actions"] = self.max_actions
            payload["ac_fixed"] = self.ac_fixed.state_dict()
        else:
            payload["ac_dyn"] = self.ac_dyn.state_dict()

        payload["rnd"] = self.rnd.state_dict() if self.rnd is not None else None
        torch.save(payload, path)

    @staticmethod
    def load(path, device="cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        meta = ckpt.get("meta", {})

        arch = meta.get("arch", "fixed")
        state_dim = int(meta.get("state_dim", 0))
        node_emb_dim = int(meta.get("node_emb_dim", 64))
        use_rnd = bool(meta.get("use_rnd", True))
        rnd_scale = float(meta.get("rnd_scale", 0.05))

        E = ckpt.get("E", None)
        node_map = ckpt.get("node_map", None)
        if E is None:
            E = np.zeros((1, node_emb_dim), dtype=np.float32)
        if node_map is None:
            node_map = {}

        if arch == "dynamic":
            agent = PPOAgent(
                E,
                node_map,
                state_dim,
                arch="dynamic",
                device=device,
                use_rnd=use_rnd,
                rnd_scale=rnd_scale,
            )
            agent.ac_dyn.load_state_dict(ckpt["ac_dyn"])
        else:
            max_actions = int(meta.get("max_actions"))
            agent = PPOAgent(
                E,
                node_map,
                state_dim,
                max_actions=max_actions,
                arch="fixed",
                device=device,
                use_rnd=use_rnd,
                rnd_scale=rnd_scale,
            )
            agent.ac_fixed.load_state_dict(ckpt["ac_fixed"])

        rnd_state = ckpt.get("rnd", None)
        if rnd_state is not None and agent.rnd is not None:
            agent.rnd.load_state_dict(rnd_state)

        agent.gat_meta = meta.get("gat_meta", {})
        return agent

    def update(
        self,
        buf,
        ppo_epochs=4,
        batch_size=1024,
        gamma=0.99,
        lam=0.95,
        clip_ratio=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=1.0,
    ):
        states = np.asarray(buf["states"], dtype=np.float32)
        actions = np.asarray(buf["actions"], dtype=np.int64)
        old_logp = np.asarray(buf["logp"], dtype=np.float32)
        values = np.asarray(buf["values"], dtype=np.float32)
        rewards = np.asarray(buf["rewards"], dtype=np.float32)

        rewards = np.clip(rewards, -100.0, 100.0)

        dones = np.asarray(buf["dones"], dtype=np.int64)
        obj_ids = buf.get("obj_ids", [])
        cand_ids_list = buf.get("cand_ids", [])
        last_value = float(buf.get("last_value", 0.0))

        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        gae = 0.0
        next_value = last_value
        for t in reversed(range(T)):
            mask = 1.0 - float(dones[t])
            delta = rewards[t] + gamma * next_value * mask - values[t]
            gae = delta + gamma * lam * mask * gae
            adv[t] = gae
            next_value = values[t]
        ret = adv + values

        if T > 0:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        if T == 0:
            return {"loss_pi": 0.0, "loss_v": 0.0, "loss_ent": 0.0}

        idxs = np.arange(T)
        last_loss_pi = 0.0
        last_loss_v = 0.0
        last_loss_ent = 0.0

        for _ in range(ppo_epochs):
            np.random.shuffle(idxs)
            for start in range(0, T, batch_size):
                mb = idxs[start : start + batch_size]
                if len(mb) == 0:
                    continue

                loss_pi = 0.0
                loss_v = 0.0
                loss_ent = 0.0

                self.optim_actor.zero_grad()
                self.optim_critic.zero_grad()

                for j in mb:
                    s_np = self._pad_or_trunc(states[j], self.state_dim)
                    s = torch.tensor(s_np, dtype=torch.float, device=self.device).unsqueeze(0)

                    idx_obj = self._idx(obj_ids[j])
                    idx_obj = max(0, min(int(idx_obj), int(self.E.shape[0]) - 1))
                    z_obj = self.E[idx_obj].unsqueeze(0)

                    cand_ids = cand_ids_list[j]
                    if len(cand_ids) == 0:
                        continue
                    cand_idx = [max(0, min(int(self._idx(cid)), int(self.E.shape[0]) - 1)) for cid in cand_ids]
                    z_cands = self.E[cand_idx].unsqueeze(0)

                    logits, v = self.ac_dyn.score(z_obj, z_cands, s)
                    probs = F.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)

                    a = torch.tensor(actions[j], device=self.device)
                    logp = dist.log_prob(a)
                    ratio = torch.exp(logp - torch.tensor(old_logp[j], device=self.device))

                    adv_t = torch.tensor(adv[j], dtype=torch.float, device=self.device)
                    ret_t = torch.tensor(ret[j], dtype=torch.float, device=self.device)

                    surr1 = ratio * adv_t
                    surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv_t
                    pi_loss = -torch.min(surr1, surr2)

                    v_loss = F.mse_loss(v.squeeze(0), ret_t)
                    ent = dist.entropy().mean()

                    (pi_loss + vf_coef * v_loss - ent_coef * ent).backward()

                    loss_pi += float(pi_loss.item())
                    loss_v += float(v_loss.item())
                    loss_ent += float(ent.item())

                clip_grad_norm_(
                    list(self.ac_dyn.scorer.parameters()) + list(self.ac_dyn.critic.parameters()),
                    max_grad_norm,
                )
                self.optim_actor.step()
                self.optim_critic.step()

                last_loss_pi = loss_pi / max(1, len(mb))
                last_loss_v = loss_v / max(1, len(mb))
                last_loss_ent = loss_ent / max(1, len(mb))

        return {"loss_pi": last_loss_pi, "loss_v": last_loss_v, "loss_ent": last_loss_ent}
