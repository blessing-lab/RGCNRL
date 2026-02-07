# src/PodemQuest/PodemQuest.py
# Apache License 2.0
#!/usr/bin/env python3
import argparse
import time
from .PODEM import PODEM
from .Circuit import Circuit


def main():
    parser = argparse.ArgumentParser(description="Run PODEM on a specified input file.")
    parser.add_argument("-i", "--input_file", type=str, required=True, help="The input file to be processed by PODEM")
    parser.add_argument("-o", "--output_file", type=str, required=True, help="The output file to save the PODEM report")
    parser.add_argument("-r", "--report_file", type=str, default=None, help="The file to save the detailed PODEM report")

    # RL（推理）
    parser.add_argument("--policy_path", type=str, default=None, help="Path to a trained PPO policy (.pt).")
    parser.add_argument("--cpu", action="store_true", help="Force running RL policy on CPU.")

    # RL 策略（默认：hybrid，优先使用在线统计，统计不足时再参考 policy）
    parser.add_argument(
        "--rl_strategy",
        type=str,
        default="hybrid",
        choices=["hybrid", "policy", "stats", "heuristic"],
        help="Backtrace fanin selection strategy when --policy_path is provided.",
    )
    parser.add_argument("--rl_min_conf", type=float, default=0.60, help="Min max-prob to trust RL logits.")
    parser.add_argument("--rl_max_entropy", type=float, default=1.2, help="Max entropy to trust RL logits.")
    parser.add_argument("--bt_stat_min_count", type=int, default=3, help="Min stats count to enable stats-based choice.")
    parser.add_argument("--bt_ucb_c", type=float, default=0.7, help="UCB exploration coefficient for stats-based choice.")
    parser.add_argument("--search_max_depth", type=int, default=50, help="Max recursion depth for advanced PODEM.")

    # 搜索控制
    parser.add_argument("--max_backtracks", type=int, default=None, help="Maximum backtracks.")
    args = parser.parse_args()

    circuit = Circuit(args.input_file)
    podem_agent = PODEM(circuit=circuit, output_file=args.output_file)

    # 将策略参数下发到 PODEM（即使不使用 RL，也可用于统计分支排序）
    podem_agent.rl_strategy = args.rl_strategy
    podem_agent.rl_min_conf = args.rl_min_conf
    podem_agent.rl_max_entropy = args.rl_max_entropy
    podem_agent.bt_stat_min_count = args.bt_stat_min_count
    podem_agent.bt_ucb_c = args.bt_ucb_c
    podem_agent.search_max_depth = args.search_max_depth

    if args.max_backtracks is not None:
        podem_agent.max_backtracks = int(args.max_backtracks)

    if args.policy_path:
        try:
            import torch
            device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")

            from SmartATPGPro.ppo_rnd_agent import PPOAgent
            from SmartATPGPro.gat_embedder import compute_embeddings

            policy = PPOAgent.load(args.policy_path, device=device)

            # 用训练时保存的 GAT 超参为“当前电路”计算 embedding，然后注入 policy
            gat_meta = getattr(policy, "gat_meta", {}) or {}
            heads = int(gat_meta.get("heads", 8))
            hidden = int(gat_meta.get("hidden", 128))
            out_dim = int(gat_meta.get("out_dim", 64))
            seed = int(gat_meta.get("seed", 0))

            # 固定 seed，保证训练/推理的 embedding 生成一致
            import numpy as np
            import random
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            emb, node_map = compute_embeddings(
                circuit, hidden=hidden, out_dim=out_dim, heads=heads, device=device
            )
            policy.set_graph_embeddings(emb, node_map)

            # 强制贪心
            original_act = policy.act

            def greedy_act(state_dict, deterministic=False):
                return original_act(state_dict, deterministic=True)

            policy.act = greedy_act

            podem_agent.policy = policy
            arch = getattr(policy, "arch", "dynamic")
            print(f"[RL] Loaded greedy policy from {args.policy_path} (device={device}, arch={arch})")

        except Exception as e:
            print(f"[RL] Failed to load policy: {e}")
            podem_agent.policy = None

    start_time = time.time()
    podem_agent.compute(algorithm="advanced")
    total_time = time.time() - start_time
    report = podem_agent.report()
    combined_report = f"""
    ================== PODEM Fault Coverage Report ==================
        {report.strip()}
    ------------------------------------------------------------------
    Total Time Taken: {total_time:.4f} seconds
    ==================================================================
    """
    if args.report_file:
        with open(args.report_file, "w") as f:
            f.write(combined_report)


if __name__ == "__main__":
    main()
