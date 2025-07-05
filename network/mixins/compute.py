from utils.softmax import softmax


class ComputeMixin:
    def compute(self, input: list[float]):
        self.set_input_layer_data(input)

        for sender, receiver in self.loop_through_layers():
            self.connect_layers(sender, receiver)

        output = softmax([node.get_output() for node in self.output_layer.nodes])

        predicted_output_map = [
            (idx, output[idx]) for idx in range(len(self.output_layer.nodes))
        ]

        sorted_predictions = sorted(
            predicted_output_map,
            key=lambda prediction: prediction[1],
            reverse=True,
        )

        confident = sorted_predictions[0]
        predicted_number = confident[-1]
        return predicted_output_map, predicted_number
