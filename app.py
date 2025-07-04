from network.network import Network
from network.controller import Controller
from network.layer import Layer
from network.trainer import Trainer

from data import x_train, y_train, x_test, y_test
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
                num_nodes=28 * 28,
            )
        )

        self.network.add_layer(
            Layer(
                name="HIDDEN_LAYER_1",
                num_nodes=64,
            )
        )

        self.network.add_layer(
            Layer(
                name="OUTPUT_LAYER",
                num_nodes=10,
            )
        )

        controller = Controller()
        Trainer(self.network, controller).train(x_train, y_train)


if __name__ == "__main__":
    app = App()
    app.run()

    export_path = os.path.join(BASE_DIR, "data", "exported_network.json")
    app.network.export(output_path=export_path)
