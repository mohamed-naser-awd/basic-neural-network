import numpy as np
from typing import Self
import uuid


class Node:
    weight: dict[str, float]
    _input: float
    id: str

    def __init__(self):
        self.id = str(uuid.uuid4().hex)
        self.weights = {}
        self._input = None

    @property
    def input(self):
        return self._input

    @input.setter
    def input(self, value: float):
        if not isinstance(value, (int, float, np.uint8)):
            raise ValueError(f"Input must be a number, got: {value}, type({type(value)}).")
        self._input = value
        return self._input

    def add_weight(self, link_name: str, weight: float):
        self.weights[link_name] = weight
        return weight

    def export(self):
        return {
            "weights": self.weights,
            "id": self.id
        }

    def get_output(self):
        return self.input

    def get_next_node_output(self, next_node: Self):
        next_node_id = next_node.id
        next_node_weight = self.weights.get(next_node_id)

        if not next_node_weight:
            next_node_weight = self.add_weight(next_node_id, np.random.rand())

        return next_node_weight * self.input
