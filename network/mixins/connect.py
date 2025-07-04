from network.layer import Layer


class LayerConnectionMixin:
    """
    Controller manage connecting nodes to each other.
    Takes output from each layer then applies relu activation function.
    Pass the activated output to the next layer.
    """

    def connect_layers(self, sender: Layer, receiver: Layer):
        receiver_nodes = receiver.nodes
        sender_nodes = sender.nodes

        for receiver_node in receiver_nodes:
            summed_input = 0  # sent to receiver node
            for sender_node in sender_nodes:
                # Assuming each node has a method to get its output
                output = sender_node.get_next_node_output(receiver_node)
                summed_input += output

            receiver_node.input = self.relu(summed_input)

    def relu(self, summed_input: float) -> float:
        """
        Apply relu activation function.
        """
        return max(0, summed_input)
