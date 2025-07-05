from utils.loss import cross_entropy_loss
from network.node import Node
from network.network import Network
import settings


class NetworkTrainer:
    """
    Does the actual network training
    """

    network: Network

    def __init__(self, network: Network):
        self.network = network

    def train(self, x_train, y_train: list[int]):
        """
        1. Preprocess the data if necessary.
        2. Iterate through the layers and train each layer.
        3. Use the training data to adjust the weights the layers.
        """
        x_train, y_train = self.network.preprocess(x_train, y_train)

        for input, output in zip(x_train, y_train):
            self.train_from_sample(input, output)

    def train_from_sample(self, input, true_output):
        predicted_output_map, predicted_number = self.network.compute(input)
        loss = cross_entropy_loss([i[1] for i in predicted_output_map], true_output)

        if loss > 0.3:
            self.adjust_output(predicted_number, true_output, predicted_output_map)

    def adjust_output(self, predicted_output, true_output, predicted_output_map):

        wrong_nodes = self.get_wrong_nodes(
            predicted_output, true_output, predicted_output_map
        )

        print(f"adjusting output, predicted: {predicted_output}, true: {true_output}, number of wrong nodes: {len(wrong_nodes)}")


        for node, dl_dz in wrong_nodes:
            self.update_output_node_weights(node, dl_dz, settings.LEARNING_RATE)

    def get_wrong_nodes(self, predicted_output, true_output, predicted_output_map):
        wrong_nodes: list[tuple[Node, float]] = []

        for idx, node in enumerate(self.network.output_layer.nodes):
            node_output, node_confidence = predicted_output_map[idx]

            correct_confidence = 1 if true_output == node_output else 0
            wrongness = node_confidence - correct_confidence

            if float(wrongness) != float(0):
                wrong_nodes.append((node, wrongness))

        return wrong_nodes

    def update_output_node_weights(self, wrong_node: Node, dL_dz, learning_rate=0.01):
        for sender_node in wrong_node.input_nodes:
            a_prev = sender_node.get_output()  # activation of sender node
            weight = sender_node.weights[wrong_node.id]

            # Gradient: dL/dw = dL/dz * a_prev
            gradient = dL_dz * a_prev

            # Update: w = w - lr * gradient
            new_weight = weight - learning_rate * gradient

            sender_node.weights[wrong_node.id] = new_weight
            sender_node.reset()

        wrong_node.reset()
