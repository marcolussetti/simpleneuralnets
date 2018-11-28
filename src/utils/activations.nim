import math

proc sigmoidActivation*(z: float): float =
    return 1.0 / (1.0 + E.pow(-z))

proc derivative*(n: float): float =
    return n * (1.0 - n)

proc sigmoidPrime*(z: float): float =
    return derivative(sigmoidActivation(z))

proc tanhActivation*(z: float): float =
    return sinh(z) / cosh(z)
