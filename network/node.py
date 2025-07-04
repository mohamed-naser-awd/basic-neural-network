import numpy as np


class Node:
    weight: dict[str, float]

    def __init__(self):
        self.weights = {}
        self.input = None

    def add_weight(self, link_name: str, weight: float):
        self.weights[link_name] = weight

    def export(self):
        return {
            "weights": self.weights,
        }
