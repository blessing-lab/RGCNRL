# src/PodemQuest/PODEM.py
# Apache License 2.0
import math
from collections import Counter
from .DAlgebra import D_Value


class PODEM:
    """带 RL 回溯选择（可选）的 PODEM。

    重要修正（影响覆盖率/回溯统计）：
    1) activate_fault 只“标记故障”，不在递归外提前 backtrace+赋PI+imply
    2) get_objective 在“未激活故障”阶段应返回 fault_value（激励目标），不是 oppositeVal(fault_value)
    3) 默认禁用 warm-start，避免把内部结点先定值导致 objective 直接 None

    训练增强：
    - 增加 collector：在 backtrace_advanced 的“第一层扇入选择”处采样动作，并在对应子树结束后用 Δbt 回填 reward
      让 PPO 直接优化“减少 PI 翻转(bt_count)”与“一步到位(Δbt=0 的成功)”。
    """

    def __init__(self, circuit, output_file):
        self.circuit = circuit
        self.output_file = output_file
        self.fault_is_activated = False
        self.D_Frontier = []
        self.fault_gate = None
        self.fault_value = None
        self.fault_stuck = None  # 0->SA0, 1->SA1（用于构造 state 特征）

        self.no_of_faults = len(self.circuit.faults)
        self.detected_faults = 0
        self.failures = 0
        self.fault_coverage = 0

        # RL policy（可选，由外部注入）
        self.policy = None

        try:
            self.max_fanin = max(
                (len(g.input_gates) for g in self.circuit.gates.values()), default=1
            )
        except Exception:
            self.max_fanin = 1

        # 回溯上限相关计数（原有语义）
        self.max_backtracks = max(1000, len(self.circuit.primary_input_gates) * 100)
        self.bt_count = 0  # 每个故障内部的回溯步数计数（用于停止条件）

        # 总回溯次数统计（整个 compute() 过程中所有故障的 bt_count 累加）
        self.total_backtracks = 0

        # gate 的历史回溯统计：gate_id -> {"count": n, "sum_bt": s}
        self.gate_bt_stats = {}

        # 默认禁用 warm-start
        self.enable_warm_start = False

        # --------- 推理期策略参数 ---------
        self.rl_strategy = "hybrid"  # {"hybrid","policy","heuristic","stats"}
        self.bt_stat_min_count = 3
        self.bt_ucb_c = 0.7
        self.rl_min_conf = 0.60
        self.rl_max_entropy = 1.2
        self.search_max_depth = 50

        # --------- 训练期采样 collector（可选）---------
        # collector 必须是 dict，包含 keys:
        #  states/actions/logp/values/rewards/dones/obj_ids/cand_ids
        self.collector = None

        # 训练 reward shaping：默认偏向“一步到位 + 少回溯”
        self.collect_bt_penalty = 1.0        # 每次 PI 翻转的惩罚系数
        self.collect_bt_cap = 50             # Δbt 截断，防极端样本
        self.collect_decision_cost = 0.05    # 每次做扇入选择的小惩罚（鼓励更短轨迹）
        self.collect_depth_cost = 0.15       # 与 depth_norm 成比例的惩罚（越深越贵）
        self.collect_zero_bt_bonus = 1.0     # 若该决策最终成功且 Δbt==0，额外奖励
        self.collect_reward_clip = 20.0      # 单步 reward 裁剪

    # -------------------- collector helpers --------------------
    def set_collector(self, collector: dict | None):
        self.collector = collector

    def _collector_add_transition(self, state_dict, action, logp, value):
        """新增一条 transition（reward/done 先占位，等子树结束回填 reward）"""
        if self.collector is None or self.policy is None:
            return None
        buf = self.collector
        idx = len(buf["rewards"])
        enc_state = self.policy.encode_state_vec(state_dict)
        buf["states"].append(enc_state)
        buf["actions"].append(int(action))
        buf["logp"].append(float(logp))
        buf["values"].append(float(value))
        buf["rewards"].append(0.0)
        buf["dones"].append(0)
        buf["obj_ids"].append(state_dict.get("node_id", state_dict.get("node", None)))
        buf["cand_ids"].append(list(state_dict.get("cand_ids", [])))
        return idx

    def _collector_finish_transition(self, decision_idx, delta_bt, depth_norm, success_subtree: bool):
        """用 Δbt 回填 reward；可附加“一步到位”奖励"""
        if self.collector is None or decision_idx is None:
            return
        bt_pen = float(self.collect_bt_penalty)
        bt_cap = float(self.collect_bt_cap)
        dec_cost = float(self.collect_decision_cost)
        dep_cost = float(self.collect_depth_cost)
        zero_bonus = float(self.collect_zero_bt_bonus)
        clipv = float(self.collect_reward_clip)

        d = float(delta_bt)
        d = max(0.0, min(d, bt_cap))

        r = -bt_pen * d - dec_cost - dep_cost * float(depth_norm)
        if success_subtree and d <= 1e-6:
            r += zero_bonus

        # 单步裁剪，避免极端子树造成不稳定
        if clipv > 0:
            r = max(-clipv, min(r, clipv))

        self.collector["rewards"][decision_idx] += float(r)

    # -------------------- RL 状态构造 --------------------
    def _ensure_graph_stats(self):
        """缓存 level / fanout / max_level / max_fanout，用于构造 RL state 特征。"""
        if hasattr(self, "_level_map") and hasattr(self, "_fanout_map"):
            return
        from collections import deque

        nodes = list(self.circuit.gates.keys())
        level = {nid: 0 for nid in nodes}

        q = deque([g.outputpin for g in self.circuit.primary_input_gates])
        seen = set(q)
        while q:
            nid = q.popleft()
            for og in self.circuit.gates[nid].output_gates:
                lid = og.outputpin
                level[lid] = max(level[lid], level[nid] + 1)
                if lid not in seen:
                    seen.add(lid)
                    q.append(lid)

        fanout = {nid: len(self.circuit.gates[nid].output_gates) for nid in nodes}

        self._level_map = level
        self._fanout_map = fanout
        self._max_level = max(level.values()) if level else 1
        self._max_fanout = max(fanout.values()) if fanout else 1

    def _build_state_for_policy(self, gate, objective_value, current_depth=0, max_depth=50):
        """state 字段（推理/训练统一）：
        node, gate_type, level, fanout, CC0, CC1, CCb, fault_bt_sa0, fault_bt_sa1, gat_vec
        + phase, obj_val, depth_norm, cand_count_norm   (新增 4 个关键字段，提升“一步到位”能力)
        """
        import numpy as np

        try:
            from SmartATPGPro.gat_embedder import GATE_TYPES
        except Exception:
            GATE_TYPES = [
                "AND", "OR", "XOR", "NOT", "NAND", "NOR", "XNOR", "BUFF", "BUF", "input_pin", "output_pin"
            ]

        self._ensure_graph_stats()

        node = gate.outputpin
        gate_type = gate.type

        level = int(getattr(self, "_level_map", {}).get(node, 0))
        fanout = int(getattr(self, "_fanout_map", {}).get(node, 0))
        max_level = float(getattr(self, "_max_level", 1.0))
        max_fanout = float(getattr(self, "_max_fanout", 1.0))
        level_norm = float(level) / (max_level + 1e-6)
        fanout_norm = float(fanout) / (max_fanout + 1e-6)

        # phase: 0=激活故障, 1=传播 D-frontier
        phase = 1.0 if bool(getattr(self, "fault_is_activated", False)) else 0.0

        # obj_val: 目标值(0/1)
        if objective_value == D_Value.ONE:
            obj_val = 1.0
        elif objective_value == D_Value.ZERO:
            obj_val = 0.0
        else:
            obj_val = 0.5

        depth_norm = float(current_depth) / float(max_depth + 1e-6)

        candidates = [g for g in gate.input_gates if g.value == D_Value.X]
        cand_ids = [c.outputpin for c in candidates]
        cand_count_norm = float(len(candidates)) / float(max(1, int(getattr(self, "max_fanin", 1))))

        mask = np.ones(len(candidates), dtype=np.float32) if len(candidates) > 0 else None

        # 当前故障类型下的“回溯次数”信号（与推理一致：用 bt_count）
        bt = float(getattr(self, "bt_count", 0))
        fault_bt_sa0 = bt if int(getattr(self, "fault_stuck", -1)) == 0 else 0.0
        fault_bt_sa1 = bt if int(getattr(self, "fault_stuck", -1)) == 1 else 0.0

        # 可选：node embedding（推理期由外部注入 policy.E/node_map）
        gat_vec = None
        if self.policy is not None and hasattr(self.policy, "E") and hasattr(self.policy, "node_map"):
            try:
                idx = int(self.policy.node_map.get(node, 0))
                idx = max(0, min(idx, int(self.policy.E.shape[0]) - 1))
                gat_vec = self.policy.E[idx].detach().cpu().numpy()
            except Exception:
                gat_vec = None

        return {
            "node": node,
            "node_id": node,
            "gate_type": gate_type,
            "level": level,
            "level_norm": level_norm,
            "fanout": fanout,
            "fanout_norm": fanout_norm,
            "max_level": max_level,
            "max_fanout": max_fanout,
            "CC0": float(gate.CC0),
            "CC1": float(gate.CC1),
            "CCb": float(gate.CCb),
            "fault_bt_sa0": float(fault_bt_sa0),
            "fault_bt_sa1": float(fault_bt_sa1),
            "phase": float(phase),
            "obj_val": float(obj_val),
            "depth_norm": float(depth_norm),
            "cand_count_norm": float(cand_count_norm),
            "gat_vec": gat_vec,
            "cand_ids": cand_ids,
            "mask": mask,
            "candidates": candidates,  # 仅本地使用
        }

    # -------------------- 主流程 --------------------
    def compute(self, algorithm="advanced"):
        self.total_backtracks = 0
        self.detected_faults = 0
        self.failures = 0

        self.circuit.calculate_SCOAP()

        if algorithm == "basic":
            for fault in self.circuit.faults:
                self.init_PODEM()
                for PI in self.circuit.primary_input_gates:
                    PI.explored = False
                self.basic_PODEM(fault)
            return

        test_lines = []
        known_vectors = []
        total_faults = len(self.circuit.faults)

        self.known_vectors = known_vectors

        for idx, fault in enumerate(self.circuit.faults):
            reused = False
            for vec in known_vectors:
                if self.circuit.pattern_detects_fault(vec, fault):
                    self.detected_faults += 1
                    test_lines.append(str(self.detected_faults) + ": " + f"{vec}\n")
                    reused = True
                    break

            if not reused:
                self.init_PODEM()
                self.activate_fault(fault)
                ret = self.advanced_PODEM()

                self.total_backtracks += self.bt_count

                if self.fault_gate is not None:
                    self.fault_gate.faulty = False

                if ret:
                    self.detected_faults += 1
                    sv = self.ret_success_vector()
                    sv = "".join(["0" if c == "X" else c for c in sv])
                    test_lines.append(str(self.detected_faults) + ": " + f"{sv}\n")
                    if sv not in known_vectors:
                        known_vectors.append(sv)
                else:
                    self.failures += 1

            print(f"idx: {idx} / {total_faults}")

        header = "* Test pattern file\n* generated by PodemQuest"
        with open(self.output_file, "w") as f:
            f.write(header + "\n")
            f.writelines(test_lines)

    # -------------------- 初始化/激活/传播 --------------------
    def init_PODEM(self):
        self.fault_is_activated = False
        self.bt_count = 0
        for g in self.circuit.gates.values():
            g.value = D_Value.X

        for PI in getattr(self.circuit, "primary_input_gates", []):
            try:
                PI.explored = False
            except Exception:
                pass

    def activate_fault(self, fault):
        """fault: (fault_site:str, stuck:int) stuck=0->SA0, stuck=1->SA1"""
        site, stuck = fault
        self.fault_stuck = int(stuck)
        self.fault_gate = self.circuit.gates[site]
        self.fault_gate.faulty = True

        if stuck == 0:
            self.fault_value = D_Value.ONE
            self.fault_gate.fault_value = D_Value.ZERO
        else:
            self.fault_value = D_Value.ZERO
            self.fault_gate.fault_value = D_Value.ONE

        self.fault_gate.value = D_Value.X
        self.fault_is_activated = False

    def imply(self, _input_gate):
        initial_output_value = _input_gate.value
        _input_gate.evaluate()
        if initial_output_value == _input_gate.value and _input_gate.type != "input_pin":
            return
        for next_gate in _input_gate.output_gates:
            self.imply(next_gate)

    # -------------------- backtrace --------------------
    def backtrace(self, objective_gate, objective_value):
        target_PI = objective_gate
        inversion_parity = target_PI.inversion_parity
        while target_PI.type != "input_pin":
            for previous_gate in target_PI.input_gates:
                if previous_gate.value == D_Value.X:
                    target_PI = previous_gate
                    break
            inversion_parity += target_PI.inversion_parity
        target_PI_value = (
            self.oppositeVal(objective_value) if (inversion_parity % 2 == 1) else objective_value
        )
        return target_PI, target_PI_value

    def check_imply_gate(self, gate, value):
        if value == D_Value.ONE:
            return not (gate.type in ["OR", "NAND"])
        elif value == D_Value.ZERO:
            return not (gate.type in ["AND", "NOR"])
        return True

    def get_easiest_to_satisfy_gate(self, objective_gate, objective_value):
        easiest_gate = None
        easiest_value = math.inf
        for gate in objective_gate.input_gates:
            if gate.value != D_Value.X:
                continue
            if objective_value == D_Value.ZERO:
                if gate.CC0 < easiest_value:
                    easiest_gate = gate
                    easiest_value = gate.CC0
            elif objective_value == D_Value.ONE:
                if gate.CC1 < easiest_value:
                    easiest_gate = gate
                    easiest_value = gate.CC1
        return easiest_gate

    def get_hardest_to_satisfy_gate(self, objective_gate, objective_value):
        hardest_gate = None
        hardest_value = -math.inf
        for gate in objective_gate.input_gates:
            if gate.value != D_Value.X:
                continue
            if objective_value == D_Value.ZERO:
                if gate.CC0 > hardest_value:
                    hardest_gate = gate
                    hardest_value = gate.CC0
            elif objective_value == D_Value.ONE:
                if gate.CC1 > hardest_value:
                    hardest_gate = gate
                    hardest_value = gate.CC1
        return hardest_gate

    def backtrace_advanced(self, objective_gate, objective_value, current_depth=0):
        """在标准 backtrace 的基础上，做扇入选择（统计/RL/启发式）。

        返回：(target_PI, target_PI_value, chosen_gate_id, decision_idx)
        - chosen_gate_id：第一层扇入 gate_id
        - decision_idx：若该次选择由 RL policy 采样产生，则返回对应 transition 的索引；否则 None
        """
        import numpy as np

        rl_strategy = getattr(self, "rl_strategy", "hybrid")
        min_hist = int(getattr(self, "bt_stat_min_count", 3))
        ucb_c = float(getattr(self, "bt_ucb_c", 0.7))
        rl_min_conf = float(getattr(self, "rl_min_conf", 0.60))
        rl_max_entropy = float(getattr(self, "rl_max_entropy", 1.2))

        target = objective_gate
        val = objective_value
        max_depth = int(getattr(self, "search_max_depth", 50))
        used_rl = False
        chosen_gate_id = None
        decision_idx = None  # collector index

        def _avg_bt_cost(gid):
            stat = self.gate_bt_stats.get(gid)
            if not stat:
                return None, 0
            c = int(stat.get("count", 0))
            if c <= 0:
                return None, 0
            s = float(stat.get("sum_bt", 0.0))
            return (s / max(1, c)), c

        def _select_by_stats(cands):
            total_n = 0
            for g in cands:
                _, c = _avg_bt_cost(g.outputpin)
                total_n += c
            if total_n < min_hist:
                return None

            best = None
            best_score = float("inf")
            logN = math.log(total_n + 1.0)
            for g in cands:
                avg, c = _avg_bt_cost(g.outputpin)
                if avg is None:
                    continue
                bonus = ucb_c * math.sqrt(logN / (c + 1.0))
                score = avg - bonus
                if score < best_score:
                    best_score = score
                    best = g
            return best

        def _select_by_policy(gate_for_state, cand_list, cur_val):
            nonlocal used_rl, decision_idx
            if self.policy is None or used_rl or len(cand_list) <= 1:
                return None

            st = self._build_state_for_policy(gate_for_state, cur_val, current_depth, max_depth)
            if not st.get("cand_ids"):
                return None

            # 训练采样：collector 存在则强制采样；推理则 deterministic=True 并做置信度门控
            try:
                if self.collector is not None:
                    a, logp, v = self.policy.act(st, deterministic=False)
                    a = int(np.clip(int(a), 0, len(cand_list) - 1))
                    decision_idx = self._collector_add_transition(st, a, logp, v)
                    used_rl = True
                    return cand_list[a]

                # 推理：使用 score_candidates 做置信度门控
                if hasattr(self.policy, "score_candidates"):
                    logits, _, probs, ent, mx = self.policy.score_candidates(st)
                    if (mx >= rl_min_conf) and (ent <= rl_max_entropy):
                        a = int(np.argmax(logits))
                        used_rl = True
                        return cand_list[max(0, min(a, len(cand_list) - 1))]
                    return None

                a, _, _ = self.policy.act(st, deterministic=True)
                a = max(0, min(int(a), len(cand_list) - 1))
                used_rl = True
                return cand_list[a]
            except Exception:
                return None

        while target.type != "input_pin":
            if getattr(target, "inversion_parity", 0):
                val = self.oppositeVal(val)

            candidates = [g for g in target.input_gates if g.value == D_Value.X]
            if not candidates:
                pi, pi_val = self.backtrace(target, val)
                return pi, pi_val, chosen_gate_id, decision_idx

            chosen = None

            # 1) stats
            if rl_strategy in ("hybrid", "stats") and self.collector is None:
                chosen = _select_by_stats(candidates)

            # 2) policy
            if chosen is None and rl_strategy in ("hybrid", "policy", "stats", "heuristic"):
                chosen = _select_by_policy(target, candidates, val)

            # 3) heuristic
            if chosen is None:
                if self.check_imply_gate(target, val):
                    nxt = self.get_hardest_to_satisfy_gate(target, val)
                else:
                    nxt = self.get_easiest_to_satisfy_gate(target, val)
                chosen = nxt if nxt is not None else candidates[0]

            if chosen_gate_id is None:
                chosen_gate_id = chosen.outputpin
            target = chosen

        return target, val, chosen_gate_id, decision_idx

    # -------------------- 目标与前沿 --------------------
    def oppositeVal(self, value):
        if value == D_Value.ZERO:
            return D_Value.ONE
        elif value == D_Value.ONE:
            return D_Value.ZERO
        return D_Value.X

    def check_error_at_primary_outputs(self):
        for output_gate in self.circuit.primary_output_gates:
            if output_gate.value in (D_Value.D, D_Value.D_PRIME):
                return True
        return False

    def ret_success_vector(self):
        return "".join(str(PI.value.value[0]) for PI in self.circuit.primary_input_gates)

    def check_D_in_circuit(self):
        for gate in self.circuit.gates.values():
            if gate.value in (D_Value.D, D_Value.D_PRIME):
                for output_gate in gate.output_gates:
                    if self.check_X_path(output_gate):
                        return True
        return False

    def check_X_path(self, gate):
        if gate.type == "output_pin":
            return True
        if gate.value == D_Value.X:
            for output_gate in gate.output_gates:
                if self.check_X_path(output_gate):
                    return True
        return False

    def generate_d_frontier(self):
        self.D_Frontier = []
        for gate in self.circuit.gates.values():
            if gate.value != D_Value.X:
                continue
            for input_gate in gate.input_gates:
                if input_gate.value in (D_Value.D, D_Value.D_PRIME):
                    if self.check_X_path(gate):
                        self.D_Frontier.append(gate)
                    break

    def get_objective(self):
        if self.fault_gate.value in (D_Value.D, D_Value.D_PRIME):
            self.fault_is_activated = True

        if not self.fault_is_activated:
            if self.fault_gate.value in (D_Value.ONE, D_Value.ZERO):
                return None, None
            return self.fault_gate, self.fault_value

        self.generate_d_frontier()
        if len(self.D_Frontier) == 0:
            return None, None
        g = min(self.D_Frontier, key=lambda gate: gate.CCb)
        for input_gate in g.input_gates:
            if input_gate.value == D_Value.X:
                return input_gate, g.non_controlling_value
        return None, None

    # -------------------- 递归求解 + 回溯上限 --------------------
    def _update_gate_bt_stats(self, gate_id, delta_bt):
        if gate_id is None:
            return
        try:
            d = float(delta_bt)
        except Exception:
            d = 0.0
        stat = self.gate_bt_stats.get(gate_id)
        if stat is None:
            stat = {"count": 0, "sum_bt": 0.0}
            self.gate_bt_stats[gate_id] = stat
        stat["count"] = int(stat.get("count", 0)) + 1
        stat["sum_bt"] = float(stat.get("sum_bt", 0.0)) + d

    def advanced_PODEM(self):
        """标准递归 PODEM +（可选）RL 扇入选择 + 在线回溯代价统计
        +（训练可选）collector：用 Δbt 回填 reward
        """
        max_depth = int(getattr(self, "search_max_depth", 50))

        def _recurse(current_depth):
            if self.check_error_at_primary_outputs():
                return True
            if self.bt_count > self.max_backtracks:
                return False
            if current_depth >= max_depth:
                return False

            objective_gate, objective_value = self.get_objective()
            if objective_gate is None:
                return False

            bt_before = self.bt_count

            target_PI, target_PI_value, chosen_gate_id, decision_idx = self.backtrace_advanced(
                objective_gate, objective_value, current_depth
            )

            target_PI.value = target_PI_value
            self.imply(target_PI)

            # 1) first try
            if _recurse(current_depth + 1):
                delta_bt = self.bt_count - bt_before
                self._update_gate_bt_stats(chosen_gate_id, delta_bt)
                self._collector_finish_transition(
                    decision_idx, delta_bt, float(current_depth) / float(max_depth + 1e-6), True
                )
                return True

            # 回溯：翻转 PI 决策并计数
            self.bt_count += 1

            target_PI.value = self.oppositeVal(target_PI_value)
            self.imply(target_PI)

            # 2) second try
            if _recurse(current_depth + 1):
                delta_bt = self.bt_count - bt_before
                self._update_gate_bt_stats(chosen_gate_id, delta_bt)
                self._collector_finish_transition(
                    decision_idx, delta_bt, float(current_depth) / float(max_depth + 1e-6), True
                )
                return True

            # 两分支都失败：完全回退
            target_PI.value = D_Value.X
            self.imply(target_PI)

            delta_bt = self.bt_count - bt_before
            self._update_gate_bt_stats(chosen_gate_id, delta_bt)
            self._collector_finish_transition(
                decision_idx, delta_bt, float(current_depth) / float(max_depth + 1e-6), False
            )
            return False

        return _recurse(0)

    # -------------------- basic_PODEM 与报告 --------------------
    def basic_PODEM(self, fault):
        self.activate_fault(fault)
        for primary_input in self.circuit.primary_input_gates:
            if getattr(primary_input, "explored", False):
                continue
            for value in [D_Value.ZERO, D_Value.ONE]:
                primary_input.explored = True
                primary_input.value = value
                self.imply(primary_input)
                if self.check_error_at_primary_outputs():
                    return True
                else:
                    if self.check_D_in_circuit():
                        break
        return False, ""

    def report(self):
        total_faults = self.no_of_faults
        self.fault_coverage = (
            0 if total_faults == 0 else (self.detected_faults / total_faults) * 100
        )

        total_cells = len(self.circuit.gates)
        gate_types = Counter([gate.type for gate in self.circuit.gates.values()])

        report_str = f"""
        Total Faults Tested     : {total_faults}
        Detected Faults         : {self.detected_faults}
        Failures                : {self.failures}
        Total Backtracks        : {self.total_backtracks}
        Fault Coverage          : {self.fault_coverage:.2f}%

        ================== Circuit Details ==================
        Total Cells             : {total_cells}
        Gate Type Breakdown     :
"""
        for gate_type, count in gate_types.items():
            report_str += f"                                  {gate_type}: {count}\n"
        return report_str
