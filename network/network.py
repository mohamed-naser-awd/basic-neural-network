from .layer import Layer
import json


class Network:
    name: str
    layers: list[Layer]

    def __init__(self, name):
        self.name = name
        self.layers = []

    def add_layer(self, layer: Layer):
        self.layers.append(layer)

    def preprocess(self, x_train: list[list[list[float]]], y_train):
        """
        1. Normalize the input data into a single flat array to feed input nodes directly.
        """
        max_length = 5
        x, y = x_train[:max_length], y_train[:max_length]

        processed_x = []

        for sample in x:
            flat_sample = []

            for sample_batch in sample:
                flat_sample += list(sample_batch)

            processed_x.append(flat_sample)

        return processed_x, y

    @property
    def input_layer(self):
        return self.layers[0]

    @property
    def output_layer(self):
        return self.layers[-1]

    @property
    def hidden_layers(self):
        return self.layers[1:]

    def set_input_layer_data(self, input: list[float]):
        for index, node in enumerate(self.input_layer.nodes):
            node.input = input[index]

    def loop_through_layers(self):
        """
        1. Iterate through each layer in the network.
        2. For each layer, connect it to the next layer using the controller.
        """
        for i in range(len(self.layers) - 1):
            sender = self.layers[i]
            receiver = self.layers[i + 1]
            yield sender, receiver

    def export(self, output_path: str):
        data = {
            "name": self.name,
            "layers": {layer.name: layer.export() for layer in self.layers},
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=4)
