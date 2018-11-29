import math

func sigmoidActivation*(z: float): float =
    return 1.0 / (1.0 + E.pow(-z))

func derivative*(n: float): float =
    return n * (1.0 - n)

func sigmoidPrime*(z: float): float =
    return derivative(sigmoidActivation(z))

func tanhActivation*(z: float): float =
    return sinh(z) / cosh(z)
