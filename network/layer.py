from .node import Node
from .enum import LayerEnum


class Layer:

    name: str
    type: LayerEnum

    def __init__(self, name, type: LayerEnum, num_nodes: int =None, nodes: list[Node] =None):
        if num_nodes is not None:
            nodes = [Node(type=type) for _ in range(num_nodes)]

        self.nodes = nodes
        self.name = name
        self.type = type

    def export(self):
        return {
            "num_nodes": len(self.nodes),
            "nodes": [node.export() for node in self.nodes],
            "type": self.type
        }
