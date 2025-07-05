from utils.softmax import softmax
from utils.relu import relu
from network.layer import Layer, LayerEnum


class LayerConnectionMixin:
    """
    Controller manage connecting nodes to each other.
    Takes output from each layer then applies activation function (relu).
    Pass the activated output to the next layer.
    """

    def connect_layers(self, sender: Layer, receiver: Layer):
        receiver_nodes = receiver.nodes
        sender_nodes = sender.nodes

        for receiver_node in receiver_nodes:
            receiver_node_inputs = []  # sent to receiver node
            receiver_node.input_nodes = sender_nodes

            for sender_node in sender_nodes:
                # Assuming each node has a method to get its output
                output = sender_node.get_next_node_output(receiver_node)
                receiver_node_inputs.append(output)

            receiver_node_input = sum(receiver_node_inputs)

            if receiver.type != LayerEnum.OUTPUT:
                receiver_node_input = relu(receiver_node_input)

            receiver_node.input = receiver_node_input
