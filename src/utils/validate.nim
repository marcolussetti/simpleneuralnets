import algorithm

proc confusionMatrix*(expected: seq[int], predicted: seq[int],
        possibleOutcomes: int): seq[seq[int]] =
    result = newSeq[seq[int]](possibleOutcomes)
    for i in 0..<possibleOutcomes:
        result[i] = newSeq[int](possibleOutcomes)
        result[i].fill(0)
    for i in 0..<expected.len:
        result[expected[i] + 1][predicted[i] + 1] += 1

proc computeAccuracy*(expected: seq[int], predicted: seq[int]): float =
    var accurate = 0
    for i in 0..<expected.len:
        if expected[i] == predicted[i]:
            inc accurate

    return accurate.float / expected.len.float
