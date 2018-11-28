import math

proc sigmoid*(z: float): float =
    return 1.0 / (1.0 + E.pow(-z))

proc derivative*(n: float): float =
    return n * (1.0 - n)

proc sigmoidPrime*(z: float): float =
    return derivative(sigmoid(z))
