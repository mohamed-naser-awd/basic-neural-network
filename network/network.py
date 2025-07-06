from .layer import Layer
from utils.softmax import softmax
import json
import settings


class Network:
    name: str
    layers: list[Layer]

    def __init__(self, name):
        self.name = name
        self.layers = []
        self.is_connected = False

    def add_layer(self, layer: Layer):
        self.layers.append(layer)

    def preprocess(self, x_train: list[list[list[float]]], y_train):
        """
        1. Normalize the input data into a single flat array to feed input nodes directly.
        """
        max_length = settings.MAX_TRAIN_SET

        if max_length:
            x, y = x_train[:max_length], y_train[:max_length]
        else:
            x, y = x_train, y_train

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

    def connect_layers(self, sender: Layer, receiver: Layer):
        receiver_nodes = receiver.nodes
        sender_nodes = sender.nodes

        for receiver_node in receiver_nodes:
            receiver_node.input_nodes = sender_nodes

            for sender_node in sender_nodes:
                # Assuming each node has a method to get its output
                sender_node.add_weight(receiver_node.id)

    def pre_compute(self):
        if not self.is_connected:
            for sender, receiver in self.loop_through_layers():
                self.connect_layers(sender, receiver)
            self.is_connected = True

    def compute(self, input: list[float]):
        self.pre_compute()

        self.set_input_layer_data(input)

        output = softmax(self.get_raw_output())

        predicted_output_map = [
            (idx, output[idx]) for idx in range(len(self.output_layer.nodes))
        ]

        sorted_predictions = sorted(
            predicted_output_map,
            key=lambda prediction: prediction[1],
            reverse=True,
        )

        confident = sorted_predictions[0]
        predicted_number = confident[0]
        return predicted_output_map, predicted_number

    def get_raw_output(self):
        output = [node.get_output() for node in self.output_layer.nodes]
        return output
