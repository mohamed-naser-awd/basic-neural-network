from .network import Network
from .controller import Controller


class Trainer:
    """
    Does the actual network training
    """

    network: Network
    controller: Controller

    def __init__(self, network, controller):
        self.network = network
        self.controller = controller

    def train(self, x_train, y_train: list[int]):
        """
        1. Preprocess the data if necessary.
        2. Iterate through the layers and train each layer.
        3. Use the training data to adjust the weights the layers.
        """
        x_train, y_train = self.network.preprocess(x_train, y_train)

        for input, output in zip(x_train, y_train):
            self.train_from_sample(input, output)

    def train_from_sample(self, input, output):
        self.network.set_input_layer_data(input)

        for sender, receiver in self.network.loop_through_layers():
            self.controller.connect_layers(sender, receiver)

        predicted_output_map = {
            node.get_output(): idx for idx, node in enumerate(self.network.output_layer.nodes)
        } # map node confidence to its corresponding number

        predicted_number = predicted_output_map[
            max(
                predicted_output_map
            )
        ]

        if predicted_number != output:
            print(f"Predicted number: {predicted_number} is not equal to actual output: {output}")

        else:
            print(f"Predection is success, number is {predicted_number}")
