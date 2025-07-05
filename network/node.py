import numpy as np
import uuid
from typing import Self
from network.enum import LayerEnum
from utils.relu import relu


class Node:
    weight: dict[str, float]
    _bias: float
    id: str
    input_nodes: list[Self]
    type: LayerEnum

    def __init__(self, type: LayerEnum, **kw):
        self.type = type
        self.id = str(uuid.uuid4().hex)
        self.weights = {}
        self.bias = np.random.rand()
        self.input_nodes = []

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

    def get_input(self):
        """
        Return the weighted sum collected from previous layer nodes
        """

        if hasattr(self, "_input"):
            return self._input

        weighted_sum = 0

        for node in self.input_nodes:
            weighted_sum += node.get_next_node_input(self)

        self._input = weighted_sum

        print(f"weighted sum is {weighted_sum}")

        return self._input

    def get_output(self):
        if self.type == LayerEnum.OUTPUT:
            return self.get_input()

        if hasattr(self, "_activation_input"):
            return self._activation_input

        self._activation_input = relu(self.get_input())
        return self._activation_input

    def add_weight(self, link_name: str, weight=None):
        if weight is None:
            weight = np.random.rand()

        self.weights[link_name] = weight

        return weight

    def export(self):
        return {"weights": self.weights, "id": self.id}

    def get_next_node_input(self, next_node: Self):
        next_node_id = next_node.id
        next_node_weight = self.weights[next_node_id]
        print(f"next node weight is: {next_node_weight}")
        return next_node_weight * self.get_output()
