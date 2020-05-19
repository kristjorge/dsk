

def mse(output, label, derivative=False):

    if not derivative:
        value = 0.5 * (output - label) ** 2
        return value

    else:
        return output - label

