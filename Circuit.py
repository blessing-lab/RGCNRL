# Apache License
# Version 2.0, January 2004
# http://www.apache.org/licenses/

from .Gate import Gate
from .DAlgebra import D_Value
import re
from collections import deque


class Circuit:

    index_id = 0

    def __init__(self, filename):
        """
        原始语义保持：解析电路 -> 建图 -> 生成故障列表。
        新增：计算拓扑序（self.topo_order），用于快速前向仿真。
        """
        self.gates = {}                  # net_name -> Gate
        self.primary_input_gates = []
        self.primary_output_gates = []
        self.get_gates_from_PI = {}
        self.faults = []
        self.topo_order = []             # 拓扑序（Gate list）

        self.parse_circuit_file(filename)
        self.generate_fault_vector()

    # ---------------------- 解析 & 建图 ----------------------
    def parse_circuit_file(self, filename):
        with open(filename, "r") as file:
            lines = file.readlines()

        input_pattern = re.compile(r"INPUT\(([\w_.\[\]0-9]+)\)")
        output_pattern = re.compile(r"OUTPUT\(([\w_.\[\]0-9]+)\)")
        gate_pattern = re.compile(r"([\w_.\[\]0-9]+) = (\w+)\(([\w_.\[\]0-9 ,]+)\)")

        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if (m := input_pattern.match(line)) is not None:
                self.add_gate("input_pin", [], str(m.group(1)).strip())
                continue

            if (m := output_pattern.match(line)) is not None:
                self.add_gate(
                    "output_pin",
                    [
                        str(m.group(1)).strip(),
                    ],
                    "output_pin_" + str(m.group(1)).strip(),
                )
                continue

            if (m := gate_pattern.match(line)) is not None:
                gate_output = str(m.group(1)).strip()
                gate_type = m.group(2).strip()
                gate_inputs = list(map(lambda x: x.strip(), m.group(3).split(",")))
                self.add_gate(gate_type, gate_inputs, gate_output)

        self.build_graph()
        self.compute_topological_order()

    def add_gate(self, type, inputs, output_pin_id):
        gate = Gate(self.index_id, type, inputs, output_pin_id)

        if type == "input_pin":
            self.primary_input_gates.append(gate)
        elif type == "output_pin":
            self.primary_output_gates.append(gate)

        self.gates[str(output_pin_id)] = gate
        self.index_id += 1

    def build_graph(self):
        for current_gate in self.gates.values():
            input_ids = current_gate.input_gates[:]
            current_gate.input_gates.clear()
            for input_id in input_ids:
                previous_gate = self.gates[input_id]
                current_gate.input_gates.append(previous_gate)
                previous_gate.output_gates.append(current_gate)

    def compute_topological_order(self):
        """
        Kahn 拓扑排序：用于线性时间前向仿真（比层层递归 imply 更稳定）。
        """
        indeg = {}
        for g in self.gates.values():
            indeg[g] = len(g.input_gates) if g.type != "input_pin" else 0

        q = deque([g for g in self.gates.values() if indeg[g] == 0])
        topo = []
        while q:
            u = q.popleft()
            topo.append(u)
            for v in u.output_gates:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)

        if len(topo) != len(self.gates):
            # 极端情况下兜底：确保所有结点都在 topo_order 中
            seen = set(topo)
            for g in self.gates.values():
                if g not in seen:
                    topo.append(g)

        self.topo_order = topo

    # ---------------------- 调试打印（保留原义） ----------------------
    def print_circuit(self):
        print("--------------------------- ---------------------------")
        for gate in self.gates.values():
            print(gate.outputpin)
            print(gate.type)
            print(gate.value)
            print()
        print("---------------------------")

    # ---------------------- 故障读入/生成（保留原义） ----------------------
    def parse_fault_file(self, fault_file):
        with open(fault_file, "r") as file:
            lines = file.readlines()
        for i in range(0, len(lines), 2):
            net_name = lines[i].strip()
            fault_value = int(lines[i + 1].strip())
            self.faults.append((net_name, fault_value))

    def generate_fault_vector(self):
        """
        与原版一致：每个网线（门输出）生成 SA0/SA1 两个故障。
        """
        for gate in self.gates.values():
            self.faults.append((gate.outputpin, 0))
            self.faults.append((gate.outputpin, 1))

    # ---------------------- SCOAP（保留原义） ----------------------
    def calculate_SCOAP(self):
        self.calculate_SCOAP_controlability()
        self.reset_explored()
        self.calculate_SCOAP_observability()

    def calculate_SCOAP_controlability(self):
        for pi in self.primary_input_gates:
            self._SCOAP_controlability_recursive(pi)

    def _SCOAP_controlability_recursive(self, gate):
        if gate.explored:
            return
        if any(not g.explored for g in gate.input_gates):
            return

        gate.calculate_CC0()
        gate.calculate_CC1()
        gate.explored = True

        for g in gate.output_gates:
            self._SCOAP_controlability_recursive(g)

    def calculate_SCOAP_observability(self):
        for po in self.primary_output_gates:
            self._SCOAP_observability_recursive(po)

    def _SCOAP_observability_recursive(self, gate):
        if gate.explored:
            return
        if any(not g.explored for g in gate.output_gates):
            return

        gate.calculate_CCb()
        gate.explored = True

        for g in gate.input_gates:
            self._SCOAP_observability_recursive(g)

    def reset_explored(self):
        for gate in self.gates.values():
            gate.explored = False

    # ---------------------- 快速前向仿真（新增，仅用于“向量复用”判定） ----------------------
    def _char_to_dval(self, ch):
        if ch == "0":
            return D_Value.ZERO
        if ch == "1":
            return D_Value.ONE
        return D_Value.X

    def apply_vector(self, vec_str):
        """
        将 PI 赋值为 vec_str（长度等于 PI 个数；字符为 '0'/'1'/'X'）
        """
        assert len(vec_str) == len(self.primary_input_gates)
        for ch, pi in zip(vec_str, self.primary_input_gates):
            pi.value = self._char_to_dval(ch)

    def reset_values_except_PIs(self):
        """
        将非 PI 的节点值重置为 X（PI 保持向量值）。
        """
        pi_set = set(self.primary_input_gates)
        for g in self.gates.values():
            if g not in pi_set:
                g.value = D_Value.X

    def _set_fault_flag(self, fault_site, stuck):
        """
        以卡死值注入故障：stuck==0 -> ZERO，stuck==1 -> ONE
        """
        fg = self.gates[fault_site]
        fg.faulty = True
        fg.fault_value = D_Value.ZERO if stuck == 0 else D_Value.ONE
        return fg

    def _clear_fault_flag(self, fg):
        fg.faulty = False
        fg.fault_value = None

    def simulate_topo(self):
        """
        按拓扑序进行一次性前向求值（门内仍用 D-代数）。
        """
        for g in self.topo_order:
            if g.type == "input_pin":
                continue
            g.evaluate()

    def pattern_detects_fault(self, vec_str, fault):
        """
        给定确定向量 vec_str（推荐全为 0/1，无 X），判断是否能在 PO 观测到 D/D'。
        该函数仅用于“向量复用”跳过 PODEM，不改变最终覆盖率判定逻辑。
        """
        # 1) 赋 PI；2) 清非 PI；3) 注入故障；4) 前向仿真；5) 观察 PO
        self.apply_vector(vec_str)
        self.reset_values_except_PIs()
        fg = self._set_fault_flag(fault[0], fault[1])
        try:
            self.simulate_topo()
            return any(po.value in (D_Value.D, D_Value.D_PRIME) for po in self.primary_output_gates)
        finally:
            self._clear_fault_flag(fg)
