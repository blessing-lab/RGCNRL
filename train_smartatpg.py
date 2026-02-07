# src/SmartATPGPro/train_smartatpg.py
"""
用两个电路（c7552 / s38584）的 top-1000 hard faults 混合训练 PPO（dynamic 动作空间）。
关键改动：
- episode 不再“手写前向流程”，而是直接跑真实 PODEM.advanced_PODEM()
- 在 PODEM.backtrace_advanced 的“第一层扇入选择”处采样动作，并在对应子树结束后用 Δbt 回填 reward
  reward 直接优化“减少 PI 翻转(bt_count)”与“一步到位(Δbt=0 的成功)”
"""
import argparse
import numpy as np
import os
import sys
import random
import torch

THIS_DIR = os.path.dirname(__file__)
SRC_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from PodemQuest.Circuit import Circuit
from PodemQuest.PODEM import PODEM

from SmartATPGPro.gat_embedder import compute_embeddings, GATE_TYPES
from SmartATPGPro.ppo_rnd_agent import PPOAgent


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_hard_faults_txt(path: str):
    faults = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            net = parts[0]
            sa = int(parts[1])
            if sa not in (0, 1):
                continue
            faults.append((net, sa))
    return faults


def run_one_fault_episode(agent: PPOAgent, circuit: Circuit, fault: tuple, buf: dict, args) -> bool:
    """
    对一个 fault 运行真实 PODEM 搜索，PODEM 内部会把每次 RL 决策的 reward 回填到 buf。
    这里仅负责：
    - 设置 terminal reward / done
    """
    podem = PODEM(circuit, output_file=os.devnull)

    # 推理/训练统一：policy 只做扇入第一层选择
    podem.policy = agent
    podem.rl_strategy = "policy"  # 训练强制走 policy（仍有 heuristic 兜底）

    # 搜索限制
    podem.search_max_depth = int(args.search_max_depth)
    if args.max_backtracks is not None:
        podem.max_backtracks = int(args.max_backtracks)

    # 训练 reward 超参
    podem.collect_bt_penalty = float(args.bt_penalty)
    podem.collect_bt_cap = float(args.bt_cap)
    podem.collect_decision_cost = float(args.decision_cost)
    podem.collect_depth_cost = float(args.depth_cost)
    podem.collect_zero_bt_bonus = float(args.zero_bt_bonus)
    podem.collect_reward_clip = float(args.reward_clip)

    podem.set_collector(buf)

    start_len = len(buf["rewards"])

    podem.init_PODEM()
    podem.activate_fault(fault)

    ok = False
    try:
        ok = bool(podem.advanced_PODEM())
    finally:
        if podem.fault_gate is not None:
            podem.fault_gate.faulty = False

    end_len = len(buf["rewards"])
    if end_len <= start_len:
        return ok  # 该故障可能没有触发任何“需要选择扇入”的决策点

    last = end_len - 1
    if ok:
        buf["rewards"][last] += float(args.success_reward)
    else:
        buf["rewards"][last] += -float(args.fail_penalty)
    buf["dones"][last] = 1
    return ok


def train(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    set_all_seeds(args.gat_seed)

    bench_c1 = os.path.join(args.bench_dir, f"{args.c1}.bench")
    bench_c2 = os.path.join(args.bench_dir, f"{args.c2}.bench")
    if not os.path.exists(bench_c1):
        raise FileNotFoundError(f"Cannot find {bench_c1}")
    if not os.path.exists(bench_c2):
        raise FileNotFoundError(f"Cannot find {bench_c2}")

    c1 = Circuit(bench_c1)
    c2 = Circuit(bench_c2)

    try:
        c1.calculate_SCOAP()
    except Exception:
        pass
    try:
        c2.calculate_SCOAP()
    except Exception:
        pass

    hf1_path = os.path.join(args.data_dir, f"{args.c1}_hard_faults.txt")
    hf2_path = os.path.join(args.data_dir, f"{args.c2}_hard_faults.txt")
    if not os.path.exists(hf1_path):
        raise FileNotFoundError(f"Cannot find hard faults file: {hf1_path}")
    if not os.path.exists(hf2_path):
        raise FileNotFoundError(f"Cannot find hard faults file: {hf2_path}")

    hard1 = load_hard_faults_txt(hf1_path)
    hard2 = load_hard_faults_txt(hf2_path)
    print(f"Loaded {len(hard1)} hard faults for {args.c1}: {hf1_path}")
    print(f"Loaded {len(hard2)} hard faults for {args.c2}: {hf2_path}")

    # 计算 embedding（固定 seed，保证训练/推理一致）
    set_all_seeds(args.gat_seed)
    emb1, node_map1 = compute_embeddings(c1, hidden=args.hidden, out_dim=args.emb, heads=args.heads, device=device)
    set_all_seeds(args.gat_seed)
    emb2, node_map2 = compute_embeddings(c2, hidden=args.hidden, out_dim=args.emb, heads=args.heads, device=device)

    # state_dim 必须与 PPOAgent._state_to_vec 对齐（新增 4 个字段）
    # onehot(11) + level(1) + fanout(1) + scoap(3) + fault_bt(2) + extra(4) = 22
    state_dim = len(GATE_TYPES) + 1 + 1 + 3 + 2 + 4

    agent = PPOAgent(
        node_embeddings=emb1,
        node_map=node_map1,
        state_dim=state_dim,
        arch="dynamic",
        device=device,
        use_rnd=not args.disable_rnd,
        rnd_scale=float(args.rnd_scale),
        lr_actor=float(args.lr_actor),
        lr_critic=float(args.lr_critic),
    )
    agent.gat_meta = {
        "heads": int(args.heads),
        "hidden": int(args.hidden),
        "out_dim": int(args.emb),
        "seed": int(args.gat_seed),
    }

    train_pairs = [(args.c1, f) for f in hard1] + [(args.c2, f) for f in hard2]

    for epoch in range(args.epochs):
        random.shuffle(train_pairs)

        buf = {k: [] for k in ["states", "actions", "logp", "values", "rewards", "dones", "obj_ids", "cand_ids"]}

        succ = 0
        total = 0

        for tag, fault in train_pairs:
            if tag == args.c1:
                agent.set_graph_embeddings(emb1, node_map1)
                circuit = c1
            else:
                agent.set_graph_embeddings(emb2, node_map2)
                circuit = c2

            if fault[0] not in circuit.gates:
                continue

            ok = run_one_fault_episode(agent, circuit, fault, buf, args)
            total += 1
            succ += 1 if ok else 0

        if len(buf["rewards"]) == 0:
            print(f"[Epoch {epoch}] No samples; skip update.")
            continue

        # ent_coef 退火：前期探索，后期更确定（更接近“查表一步到位”）
        if args.epochs <= 1:
            ent_coef = float(args.ent_coef_end)
        else:
            t = float(epoch) / float(args.epochs - 1)
            ent_coef = float(args.ent_coef_start) * (1.0 - t) + float(args.ent_coef_end) * t

        buf["last_value"] = 0.0
        stats = agent.update(
            buf,
            ppo_epochs=args.ppo_epochs,
            batch_size=args.batch_size,
            gamma=args.gamma,
            lam=args.lam,
            clip_ratio=args.clip_ratio,
            vf_coef=args.vf_coef,
            ent_coef=ent_coef,
            max_grad_norm=args.max_grad_norm,
        )

        avg_r = float(np.mean(buf["rewards"]))
        sr = 0.0 if total == 0 else (100.0 * float(succ) / float(total))
        print(f"[Epoch {epoch+1}/{args.epochs}] success_rate={sr:.2f}% avg_reward={avg_r:.4f} ent_coef={ent_coef:.5f} update={stats}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    agent.save(args.out, save_graph=False)
    print(f"[train] Saved policy to: {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bench_dir", type=str, default="test", help="Directory containing *.bench")
    p.add_argument("--data_dir", type=str, default="data", help="Directory containing *_hard_faults.txt")
    p.add_argument("--c1", type=str, default="c7552")
    p.add_argument("--c2", type=str, default="s38584")

    p.add_argument("--out", type=str, default="models/ppo_c7552_s38584.pt")
    p.add_argument("--epochs", type=int, default=25)

    # GAT embedder 参数（必须与推理一致）
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--emb", type=int, default=64)
    p.add_argument("--gat_seed", type=int, default=0)

    # 搜索限制
    p.add_argument("--search_max_depth", type=int, default=100)
    p.add_argument("--max_backtracks", type=int, default=4000)

    # PPO 超参
    p.add_argument("--ppo_epochs", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip_ratio", type=float, default=0.2)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # entropy 退火（更利于后期“确定性查表”）
    p.add_argument("--ent_coef_start", type=float, default=0.02)
    p.add_argument("--ent_coef_end", type=float, default=0.002)

    # 学习率（建议比原版更稳）
    p.add_argument("--lr_actor", type=float, default=1e-4)
    p.add_argument("--lr_critic", type=float, default=3e-4)

    # 终止奖励（建议中等尺度即可，核心由 -Δbt 主导）
    p.add_argument("--success_reward", type=float, default=60.0)
    p.add_argument("--fail_penalty", type=float, default=80.0)

    # Δbt shaping（关键）
    p.add_argument("--bt_penalty", type=float, default=1.5)
    p.add_argument("--bt_cap", type=float, default=50.0)
    p.add_argument("--decision_cost", type=float, default=0.05)
    p.add_argument("--depth_cost", type=float, default=0.15)
    p.add_argument("--zero_bt_bonus", type=float, default=1.5)
    p.add_argument("--reward_clip", type=float, default=20.0)

    # RND（建议默认关闭追求稳定/少回溯）
    p.add_argument("--disable_rnd", action="store_true", help="Disable RND intrinsic reward.")
    p.add_argument("--rnd_scale", type=float, default=0.0)

    p.add_argument("--cpu", action="store_true", help="Force CPU")

    args = p.parse_args()
    train(args)
