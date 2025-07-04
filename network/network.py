from .layer import Layer
import json


class Network:
    name: str
    layers: dict[str, Layer]

    def __init__(self, name):
        self.name = name
        self.layers = {}

    def add_layer(self, layer: Layer):
        self.layers[layer.name] = layer

    def train(self, x_train_7):
        print(f"Training network '{self.name}' with {len(x_train_7)} samples.")

    def export(self, output_path: str):
        data = {
            "name": self.name,
            "layers": {
                layer_name: layer.export() for layer_name, layer in self.layers.items()
            },
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=4)
