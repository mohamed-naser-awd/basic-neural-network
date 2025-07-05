import math


def cross_entropy_loss(y_pred, y_true):
    """
    y_pred: list of predicted probabilities (after softmax), e.g. [0.01, ..., 0.75, ...]
    y_true: index of the correct class, e.g. 9 if the true digit is 9
    """

    epsilon = 1e-15  # to avoid log(0)
    y_pred_clipped = [max(min(p, 1 - epsilon), epsilon) for p in y_pred]
    return -math.log(y_pred_clipped[y_true])
