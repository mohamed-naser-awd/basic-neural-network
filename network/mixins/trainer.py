from utils.loss import cross_entropy_loss


class TrainerMixin:
    """
    Does the actual network training
    """

    def train(self, x_train, y_train: list[int]):
        """
        1. Preprocess the data if necessary.
        2. Iterate through the layers and train each layer.
        3. Use the training data to adjust the weights the layers.
        """
        x_train, y_train = self.preprocess(x_train, y_train)

        for input, output in zip(x_train, y_train):
            self.train_from_sample(input, output)

    def train_from_sample(self, input, output):
        predicted_output_map, predicted_number = self.compute(input)

        if predicted_number != output:
            self.adjust_output(predicted_number, output, predicted_output_map)

    def adjust_output(self, predicted_output, true_output, predicted_output_map):
        print(f"adjusting output, predicted: {predicted_output}, true: {true_output}")
        loss = cross_entropy_loss([i[1] for i in predicted_output_map], true_output)
        wrong_nodes = self.get_wrong_nodes(
            predicted_output, true_output, predicted_output_map
        )

    def get_wrong_nodes(self, predicted_output, true_output, predicted_output_map):
        wrong_nodes = []

        for idx, node in enumerate(self.output_layer.nodes):
            node_output, node_confidence = predicted_output_map[idx]

            correct_confidence = 1 if true_output == node_output else 0
            wrongness = node_confidence - correct_confidence

            if float(wrongness) != float(0):
                wrong_nodes.append((node, wrongness))

        return wrong_nodes
