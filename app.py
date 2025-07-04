from network.network import Network
from network.layer import Layer
from data import x_train_7
import pathlib
import os

BASE_DIR = pathlib.Path(__file__).parent


class App:
    def __init__(self):
        self.network = Network("NUMBER_SEVEN_NETWORK")

    def run(self):
        self.network.add_layer(
            Layer(
                name="INPUT_LAYER",
                num_nodes=3,
            )
        )

        self.network.add_layer(
            Layer(
                name="HIDDEN_LAYER_1",
                num_nodes=3,
            )
        )

        self.network.add_layer(
            Layer(
                name="OUTPUT_LAYER",
                num_nodes=3,
            )
        )

        self.network.train(x_train_7)

        print(f"Network '{self.network.name}' has been trained with {len(x_train_7)} samples.")


if __name__ == "__main__":
    app = App()
    app.run()

    export_path = os.path.join(BASE_DIR, "data", "exported_network.json")
    app.network.export(output_path=export_path)
