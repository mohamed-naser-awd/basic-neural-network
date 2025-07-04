from utils.softmax import softmax

class ComputeMixin:
    def compute(self, input: list[float]):
        self.set_input_layer_data(input)

        for sender, receiver in self.loop_through_layers():
            self.connect_layers(sender, receiver)


        output = softmax(
            [
                node.get_output() for node in self.output_layer.nodes
            ]
        )

        predicted_output_map = {
            output[idx]: idx for idx in range(len(self.output_layer.nodes))
        }

        predicted_number = predicted_output_map[max(predicted_output_map)]

        return predicted_number
