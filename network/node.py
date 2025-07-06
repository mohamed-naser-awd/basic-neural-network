import numpy as np
import uuid
from typing import Self
from network.enum import LayerEnum
from utils.relu import relu


class Node:
    weights: dict[str, float]
    _bias: float
    id: str
    input_nodes: list[Self]
    type: LayerEnum
    input: float

    def __init__(self, type: LayerEnum, **kw):
        self.type = type
        kw.setdefault("id", str(uuid.uuid4().hex))
        kw.setdefault("weights", {})
        kw.setdefault("bias", np.random.rand())
        kw.setdefault("input_nodes", [])

        for k, v in kw.items():
            setattr(self, k, v)

    @property
    def bias(self):
        return self._bias

    @bias.setter
    def bias(self, value: float):
        try:
            value = float(value)
        except Exception as e:
            raise ValueError(
                f"Bias must be a number, got: {value}, type({type(value)})."
            )

        self._bias = value
        return self._bias

    def get_hidden_node_input(self):
        weighted_sum = 0

        for node in self.input_nodes:
            weighted_sum += node.get_next_node_input(self)

        return weighted_sum

    def get_input(self):
        """
        Return the weighted sum collected from previous layer nodes
        If its an input node: just return the input
        """

        if hasattr(self, "_input"):
            return self._input

        weighted_sum = 0

        if self.type == LayerEnum.INPUT:
            weighted_sum += self.input

        else:
            weighted_sum += self.get_hidden_node_input()

        self._input = weighted_sum

        return self._input

    def get_output(self):
        if self.type in [LayerEnum.OUTPUT, LayerEnum.INPUT]:
            return self.get_input()

        if hasattr(self, "_activation_output"):
            return self._activation_output

        self._activation_output = relu(self.get_input())
        return self._activation_output

    def add_weight(self, link_name: str, weight=None):
        if weight is None:
            weight = np.random.rand()

        self.weights.setdefault(link_name, weight)

        return weight

    def export(self):
        return {
            "weights": self.weights,
            "id": self.id,
            "bias": self.bias,
        }

    def get_next_node_input(self, next_node: Self):
        next_node_id = next_node.id
        next_node_weight = self.weights[next_node_id]
        res = next_node_weight * self.get_output()
        return res

    def reset(self):
        if hasattr(self, "_activation_output"):
            del self._activation_output

        if hasattr(self, "_input") and self.type != LayerEnum.INPUT:
            del self._input
