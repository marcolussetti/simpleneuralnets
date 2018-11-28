import random/urandom, random/mersenne
import alea

var rng = wrap(initMersenneTwister(urandom(16)))
let g = gaussian(mu = 0, sigma = 1)

proc generateNormalRandom*(): float =
    return rng.sample(g)

proc generateNormalRandomSequence*(length: int): seq[float] =
    for i in 0..<length:
        result.add(generateNormalRandom())

proc generateNormalRandomMatrix*(rows: int, cols: int): seq[seq[float]] =
    for i in 0..<rows:
        result.add(generateNormalRandomSequence(cols))
