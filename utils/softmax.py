import math

def softmax(logits):
    # logits: list of raw scores from output layer (z values)
    
    # Step 1: For numerical stability, subtract max from each value
    max_logit = max(logits)
    exp_values = [math.exp(x - max_logit) for x in logits]

    # Step 2: Calculate the sum of exponentials
    sum_exp = sum(exp_values)

    # Step 3: Divide each exp by the total sum
    probabilities = [exp_val / sum_exp for exp_val in exp_values]

    return probabilities
