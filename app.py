from network.network import Network
from network.layer import Layer, LayerEnum

from data import x_train, y_train
import os
import settings


class App:
    def __init__(self):
        self.network = Network("NUMBER_SEVEN_NETWORK")

    def run(self):
        self.network.add_layer(
            Layer(
                name="INPUT_LAYER",
                num_nodes=28 * 28,
                type=LayerEnum.INPUT,
            )
        )

        self.network.add_layer(
            Layer(
                name="HIDDEN_LAYER_1",
                num_nodes=64,
                type=LayerEnum.HIDDEN,
            )
        )

        self.network.add_layer(
            Layer(
                name="OUTPUT_LAYER",
                num_nodes=10,
                type=LayerEnum.OUTPUT,
            )
        )

        self.network.train(x_train, y_train)


if __name__ == "__main__":
    app = App()
    app.run()

    export_folder = os.path.join(
        settings.BASE_DIR,
        "data",
    )
    os.makedirs(export_folder, exist_ok=True)

    export_path = os.path.join(export_folder, "exported_network.json")
    app.network.export(output_path=export_path)
