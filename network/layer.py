from .node import Node
from .enum import LayerEnum


class Layer:

    name: str
    type: LayerEnum

    def __init__(self, name, num_nodes, type: LayerEnum):
        self.nodes = [Node(type=type) for _ in range(num_nodes)]
        self.name = name
        self.type = type

    def export(self):
        return {
            "num_nodes": len(self.nodes),
            "nodes": [node.export() for node in self.nodes],
        }
