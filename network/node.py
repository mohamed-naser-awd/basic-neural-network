import numpy as np
import uuid
from typing import Self
from utils.relu import relu
from network.enum import LayerEnum


class Node:
    weight: dict[str, float]
    _input: float
    _bias: float
    id: str
    connected_nodes: list[Self]
    activation_output: float
    type: LayerEnum

    def __init__(self, type: LayerEnum):
        self.type = type
        self.id = str(uuid.uuid4().hex)
        self.weights = {}
        self._input = None
        self.bias = np.random.rand()
        self.connected_nodes = []

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

    @property
    def input(self):
        return self._input

    @input.setter
    def input(self, value: float):
        try:
            value = float(value)
        except Exception as e:
            raise ValueError(
                f"Input must be a number, got: {value}, type({type(value)})."
            )
        self._input = value
        return self._input

    def add_weight(self, link_name: str, weight: float):
        self.weights[link_name] = weight
        return weight

    def export(self):
        return {"weights": self.weights, "id": self.id}

    def get_output(self):
        return self.input

    def get_next_node_output(self, next_node: Self):
        next_node_id = next_node.id
        next_node_weight = self.weights.get(next_node_id)

        if not next_node_weight:
            next_node_weight = self.add_weight(next_node_id, np.random.rand())

        return next_node_weight * self.input
