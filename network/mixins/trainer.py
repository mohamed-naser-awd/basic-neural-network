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
        predicted_number = self.compute(input)

        if predicted_number != output:
            print(
                f"Predicted number: {predicted_number} is not equal to actual output: {output}"
            )

        else:
            print(f"Predection is success, number is {predicted_number}")
