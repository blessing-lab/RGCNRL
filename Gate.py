# src/PodemQuest/Gate.py
from .DAlgebra import D_Value


class Gate:
    def __init__(self, id, type, input_gates, outputpin):
        self.id = id
        self.type = type
        self.input_gates = input_gates[:] if isinstance(input_gates, list) else []
        self.output_gates = []
        self.outputpin = outputpin
        self.value = D_Value.X
        self.faulty = False
        self.fault_value = None

        self.is_pin = (type == "input_pin" or type == "output_pin")

        # 反相奇偶
        if type in ["NOT", "NAND", "NOR", "XNOR"]:
            self.inversion_parity = 1
        else:
            self.inversion_parity = 0

        # 非控制值
        if type in ["BUFF", "BUF", "NOT"]:
            self.non_controlling_value = D_Value.ONE
        elif type in ["OR", "NOR", "XOR", "XNOR"]:
            self.non_controlling_value = D_Value.ZERO
        elif type in ["AND", "NAND"]:
            self.non_controlling_value = D_Value.ONE
        else:
            self.non_controlling_value = D_Value.X

        self.explored = False

        # 距离与 SCOAP
        self.PI_distance = 0
        self.PO_distance = 0
        self.CC0 = 0
        self.CC1 = 0
        self.CCb = 0

        # 输出可控性（修复永真 elif）
        self.is_zero_out_controllable = False
        self.is_one_out_controllable = False
        if self.type in ["AND", "NOR", "XNOR"]:
            self.is_zero_out_controllable = True
            self.is_one_out_controllable = False
        elif self.type in ["NOT", "BUFF", "BUF"]:
            self.is_zero_out_controllable = True
            self.is_one_out_controllable = True
        else:
            self.is_zero_out_controllable = False
            self.is_one_out_controllable = True

    # -------------------- 工具 --------------------
    @staticmethod
    def _invert_dval(v):
        if v == D_Value.D:
            return D_Value.D_PRIME
        if v == D_Value.D_PRIME:
            return D_Value.D
        if v == D_Value.ONE:
            return D_Value.ZERO
        if v == D_Value.ZERO:
            return D_Value.ONE
        return D_Value.X

    # -------------------- 逻辑计算 --------------------
    def evaluate(self):
        if self.type == "input_pin":
            pass
        elif self.type == "output_pin":
            self.value = self.input_gates[0].value
        elif self.type == "AND":
            self.value = self.evaluate_and()
        elif self.type == "OR":
            self.value = self.evaluate_or()
        elif self.type == "XOR":
            self.value = self.evaluate_xor()
        elif self.type in ["BUFF", "BUF"]:
            self.value = self.evaluate_buff()
        elif self.type == "NOT":
            self.value = self.evaluate_not()
        elif self.type == "NAND":
            self.value = self.evaluate_nand()
        elif self.type == "NOR":
            self.value = self.evaluate_nor()
        elif self.type == "XNOR":
            self.value = self.evaluate_xnor()

        # 故障叠加（与你原编码保持一致：取 good/fault 的第二位规则）
        if self.faulty and self.fault_value is not None:
            tempval = [self.value.value[1], self.fault_value.value[1]]
            if tempval == [1, 0]:
                self.value = D_Value.D
            elif tempval == [0, 1]:
                self.value = D_Value.D_PRIME
            elif tempval == [0, 0]:
                self.value = D_Value.ZERO
            elif tempval == [1, 1]:
                self.value = D_Value.ONE
            else:
                self.value = D_Value.X

    def evaluate_and(self):
        vals = [g.value for g in self.input_gates]
        if D_Value.ZERO in vals:
            return D_Value.ZERO
        if D_Value.X in vals:
            return D_Value.X
        if D_Value.D in vals and D_Value.D_PRIME in vals:
            return D_Value.ZERO
        if D_Value.D in vals:
            if all(v in [D_Value.ONE, D_Value.D] for v in vals):
                return D_Value.D
        if D_Value.D_PRIME in vals:
            if all(v in [D_Value.ONE, D_Value.D_PRIME] for v in vals):
                return D_Value.D_PRIME
        return D_Value.ONE

    def evaluate_or(self):
        vals = [g.value for g in self.input_gates]
        if D_Value.ONE in vals:
            return D_Value.ONE
        if D_Value.X in vals:
            return D_Value.X
        if D_Value.D in vals and D_Value.D_PRIME in vals:
            return D_Value.ONE
        if D_Value.D in vals:
            if any(v in [D_Value.ONE, D_Value.D] for v in vals):
                return D_Value.D
        if D_Value.D_PRIME in vals:
            if any(v in [D_Value.ONE, D_Value.D_PRIME] for v in vals):
                return D_Value.D_PRIME
        return D_Value.ZERO

    def evaluate_xor(self):
        vals = [g.value for g in self.input_gates]
        d = vals.count(D_Value.D)
        dp = vals.count(D_Value.D_PRIME)
        one = vals.count(D_Value.ONE)
        x = vals.count(D_Value.X)
        if x > 0:
            return D_Value.X
        # 若 D 与 D' 奇偶不等，等价于普通布尔 XOR
        if d % 2 != dp % 2:
            return D_Value.ONE if (one % 2 == 0) else D_Value.ZERO
        # 否则由 D/D' 优势决定
        if one % 2 == 0:
            return D_Value.D if d > dp else D_Value.D_PRIME
        else:
            return D_Value.D_PRIME if d > dp else D_Value.D

    def evaluate_not(self):
        v = self.input_gates[0].value
        return self._invert_dval(v)

    def evaluate_buff(self):
        return self.input_gates[0].value

    def evaluate_nand(self):
        return self._invert_dval(self.evaluate_and())

    def evaluate_nor(self):
        return self._invert_dval(self.evaluate_or())

    def evaluate_xnor(self):
        return self._invert_dval(self.evaluate_xor())

    # -------------------- SCOAP --------------------
    def calculate_CC0(self):
        if self.type == "AND":
            self.CC0 = min(g.CC0 for g in self.input_gates) + 1
        elif self.type == "NAND":
            self.CC0 = sum(g.CC1 for g in self.input_gates) + 1
        elif self.type == "OR":
            self.CC0 = sum(g.CC0 for g in self.input_gates) + 1
        elif self.type == "NOR":
            self.CC0 = min(g.CC1 for g in self.input_gates) + 1
        elif self.type == "XOR":
            self.CC0 = min(
                self.input_gates[0].CC0 + self.input_gates[1].CC0,
                self.input_gates[0].CC1 + self.input_gates[1].CC1,
            ) + 1
        elif self.type == "XNOR":
            self.CC0 = min(
                self.input_gates[0].CC1 + self.input_gates[1].CC0,
                self.input_gates[0].CC0 + self.input_gates[1].CC1,
            ) + 1
        elif self.type == "NOT":
            self.CC0 = self.input_gates[0].CC1 + 1
        elif self.type in ["BUFF", "BUF"]:
            self.CC0 = self.input_gates[0].CC0 + 1
        elif self.type == "input_pin":
            self.CC0 = 1
        elif self.type == "output_pin":
            self.CC0 = min(g.CC0 for g in self.input_gates)

    def calculate_CC1(self):
        if self.type == "AND":
            self.CC1 = sum(g.CC1 for g in self.input_gates) + 1
        elif self.type == "NAND":
            self.CC1 = min(g.CC0 for g in self.input_gates) + 1
        elif self.type == "OR":
            self.CC1 = min(g.CC1 for g in self.input_gates) + 1
        elif self.type == "NOR":
            self.CC1 = sum(g.CC0 for g in self.input_gates) + 1
        elif self.type == "XOR":
            self.CC1 = min(
                self.input_gates[0].CC0 + self.input_gates[1].CC1,
                self.input_gates[0].CC1 + self.input_gates[1].CC0,
            ) + 1
        elif self.type == "XNOR":
            self.CC1 = min(
                self.input_gates[0].CC0 + self.input_gates[1].CC0,
                self.input_gates[0].CC1 + self.input_gates[1].CC1,
            ) + 1
        elif self.type == "NOT":
            self.CC1 = self.input_gates[0].CC0 + 1
        elif self.type in ["BUFF", "BUF"]:
            self.CC1 = self.input_gates[0].CC1 + 1
        elif self.type == "input_pin":
            self.CC1 = 1
        elif self.type == "output_pin":
            self.CC1 = min(g.CC1 for g in self.input_gates)

    def calculate_CCb(self):
        CCb_output = 0
        if self.output_gates:
            CCb_output = min(g.CCb for g in self.output_gates)
        if self.type in ["AND", "NAND"]:
            self.CCb = CCb_output + sum(g.CC1 for g in self.input_gates) + 1
        elif self.type in ["OR", "NOR"]:
            self.CCb = CCb_output + sum(g.CC0 for g in self.input_gates) + 1
        elif self.type in ["NOT", "BUFF", "BUF"]:
            self.CCb = CCb_output + 1
        elif self.type == "input_pin":
            self.CCb = CCb_output
        elif self.type == "output_pin":
            self.CCb = 0
        else:
            # XOR/XNOR：此处可进一步细化，这里保持不变
            self.CCb = CCb_output + 1

    # -------------------- 其他工具 --------------------
    def check_controllable_value(self, value):
        if value == D_Value.ONE:
            return self.is_one_out_controllable
        if value == D_Value.ZERO:
            return self.is_zero_out_controllable
        return False

    def get_easiest_to_satisfy_gate(self, objective_value):
        easiest_gate = None
        easiest_value = float("inf")
        for gate in self.input_gates:
            if objective_value == D_Value.ZERO:
                if gate.CC0 < easiest_value:
                    easiest_gate, easiest_value = gate, gate.CC0
            elif objective_value == D_Value.ONE:
                if gate.CC1 < easiest_value:
                    easiest_gate, easiest_value = gate, gate.CC1
        return easiest_gate

    def get_hardest_to_satisfy_gate(self, objective_value):
        hardest_gate = None
        hardest_value = -float("inf")
        for gate in self.input_gates:
            if objective_value == D_Value.ZERO:
                if gate.CC0 > hardest_value:
                    hardest_gate, hardest_value = gate, gate.CC0
            elif objective_value == D_Value.ONE:
                if gate.CC1 > hardest_value:
                    hardest_gate, hardest_value = gate, gate.CC1
        return hardest_gate
