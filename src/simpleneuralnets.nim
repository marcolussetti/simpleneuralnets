import algorithm, math, random, sequtils, sugar, strutils
import mnist_tools
import ./utils/utils, ./utils/gaussian_random, ./utils/activations,
        ./utils/validate

export maxIndex, unzip, `/`, confusionMatrix, computeAccuracy

type
    Neuron = tuple[
            weights: seq[float], bias: float, output: float, delta: float
    ]
    Layer = seq[Neuron]
    Network* = seq[Layer]

proc newNeuron(prevLayerLength: int): Neuron =
    return (weights: generateNormalRandomSequence(prevLayerLength),
            bias: generateNormalRandom(), output: 0.0, delta: 0.0)

proc newLayer(layerLength: int, prevLayerLength: int): Layer =
    return lc[newNeuron(prevLayerLength) | (x <- 0..<layerLength), Neuron]

proc newNetwork*(dimensions: openArray[int]): Network =
    for i in 1..<dimensions.len:
        result.add(newLayer(dimensions[i], dimensions[i - 1]))

proc activation(weights: seq[float], inputs: seq[float], bias: float): float =
    result = bias             # Add bias
    for i in 0..<weights.len:
        result += weights[i] * inputs[i]

proc forwardPropagationReadOnly(network: Network, input: seq[float]): seq[
                float] =
    var inputs = input
    for i in 0..<network.len:
        var nextInputs = newSeq[float]()
        for j in 0..<network[i].len:
            let activationResult = sigmoid(activation(network[i][
                                        j].weights,
                    inputs,
                    network[i][j].bias))
            nextInputs.add(activationResult)
        inputs = nextInputs
    return inputs

proc forwardPropagation(network: var Network, input: seq[float]): seq[float] =
    var inputs = input
    for i in 0..<network.len:
        var nextInputs = newSeq[float]()
        for j in 0..<network[i].len:
            let w = network[i][j].weights
            let b = network[i][j].bias
            let sigmoidResult = sigmoid(activation(w, inputs, b))
            network[i][j].output = sigmoidResult
            nextInputs.add(network[i][j].output)
        inputs = nextInputs
    return inputs

proc backpropagation(network: var Network, expectedLabels: seq[float]) =
    var deltas = newSeq[seq[float]]()
    for i in countdown(network.high, 0):
        var errors = newSeq[float]()
        for j in 0..<network[i].len:
            # If it's the output layer, no need to sum
            if i == network.high:
                errors.add(expectedLabels[j] - network[i][
                                                j].output)
            else:
                var error = 0.0
                for neutron in network[i + 1]:
                    error += neutron.weights[
                                                        j] * neutron.delta
                errors.add(error)
        for j in 0..<network[i].len:
            network[i][j].delta = errors[j] * derivative(network[
                                        i][j].output)

proc updateNetworkWeights(network: var Network, inputs: seq[float],
    alpha: float) =
    var workingInputs = inputs
    for i in 0..<network.len:
        if i != 0:
            workingInputs = newSeq[float]()
            for neuron in network[i - 1]:
                workingInputs.add(neuron.output)
        for j in 0..<network[i].len:
            for k in 0..<workingInputs.len:
                network[i][j].weights[k] += alpha * network[
                                                i][j].delta * workingInputs[k]
            network[i][j].bias += alpha * network[i][j].delta

proc sumOfSquaredErrors(expected: seq[float], obtained: seq[float]): float =
    var sum = 0.0
    for i in 0..<expected.len:
        sum += pow(expected[i] - obtained[i], 2.0)
    return sum

proc trainBNN*(network: var Network, trainingData: seq[seq[float]],
    trainingLabels: seq[int], alpha: float, epochs: int) =
    for epoch in 0..<epochs:
        var sumErrors = 0.0
        for i in 0..<trainingData.len:
            let outputs = forwardPropagation(network,
                                        trainingData[i])
            var expected = newSeq[float](network[
                                        network.high].len)
            expected.fill(0.0)
            expected[trainingLabels[i]] = 1.0
            sumErrors += sumOfSquaredErrors(expected, outputs)
            backpropagation(network, expected)
            updateNetworkWeights(network, trainingData[i], alpha)
        echo("Epoch: ", epoch, " Error: ", sumErrors)

proc classify*(network: Network, input: seq[float]): seq[float] =
    return forwardPropagationReadOnly(network, input)

proc singleOutcome*(results: seq[float], threshold = 0.5): int =
    let maximum = maxIndex(results)
    if results[maximum] > threshold:
        return maximum
    else:
        return -1
