from network.network import Network
from network.layer import Layer
from network.enum import LayerEnum
from network.node import Node

import os
import settings


def create_network_from_file(data: dict):
    network_name = data["name"]
    network = Network(network_name)

    for idx, (layer_name, layer_data) in enumerate(data["layers"].items()):
        num_nodes = layer_data["num_nodes"]
        layer_type = layer_data["type"]
        layer_data_nodes = layer_data["nodes"]


        input_nodes = []

        if layer_type != LayerEnum.INPUT:
            input_nodes = network.layers[idx - 1].nodes

        layer_nodes = []

        for raw_node_data in layer_data_nodes:
            layer_node = Node(
                layer_type,
                **raw_node_data,
                input_nodes=input_nodes,
            )
            layer_node.reset()
            layer_nodes.append(layer_node)

        layer = Layer(layer_name, layer_type, nodes=layer_nodes)
        network.add_layer(layer)

    export_folder = os.path.join(
        settings.BASE_DIR,
        "data",
    )
    os.makedirs(export_folder, exist_ok=True)

    export_path = os.path.join(export_folder, "exported_cached_network.json")
    network.export(output_path=export_path)

    return network
