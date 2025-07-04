from .node import Node
import numpy as np


class Layer:
    name: str

    def __init__(self, name, num_nodes):
        self.nodes = [Node() for _ in range(num_nodes)]
        self.name = name

    def export(self):
        return {
            "num_nodes": len(self.nodes),
            "nodes": [
                node.export() for node in self.nodes
            ],
        }
