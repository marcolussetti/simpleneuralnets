# Simple Neural Nets

## Installation

```sh
nimble install https://github.com/marcolussetti/simpleneuralnets
```


## API
This library relies on 3 types to define a neural network:

```nim
type
    Neuron = tuple[
            weights: seq[float], bias: float, output: float, delta: float
    ]
    Layer = seq[Neuron]
Network* = seq[Layer]
```

It provides a handful of functions to interact with the network:

```nim
# Creates a new neural network initialized with random weights
proc newNetwork*(dimensions: openArray[int]): Network =

# Accepts a neural network, some training data and training labels and hyperparameters
# Note that trainingLabels are treated as input for one-hot encoding, so it indicates
# the Nth neuron should be the one indicating such an output.
# Activation function must take a single parameter z: float and return float.
# Prebuilt activation functions are sigmoidActivation and tanhActivation.
# silent attempts to suppress most output
proc trainBNN*(
    network: var Network, trainingData: seq[seq[float]],
    trainingLabels: seq[int], alpha: float, epochs: int,
    activationFunction = sigmoidActivation, silent = false
)

# Takes one input and classifies with the trained neural network
proc classify*(network: Network, input: seq[float]): seq[float]

# Takes the output of the classify function and a minimum threshold,
# and attempts to return the most likely result to pass threshold, or -1.
proc singleOutcome*(results: seq[float], threshold = 0.5): int

# Compute the accuracy in a sequence of predictions and expected values.
# Accept input already parsed with singleOutcome to reverse one-hot encoding.
proc computeAccuracy*(expected: seq[int], predicted: seq[int]): float

# Produces a confusion matrix from the predicted and expected values.
# Necessitates of a possibleOutcomes value to compute the table
proc confusionMatrix*(
    expected: seq[int], predicted: seq[int], possibleOutcomes: int
): seq[seq[int]]
```

## Examples

### Theoretical idea

Install:

```sh
nimble install https://github.com/marcolussetti/simpleneuralnets
```

Code:
```nim
import simpleneuralnets

var network = newNetwork(@[inputLayerLength, hiddenLayer1Length, hiddenLayer2Length, outputLayerLength]) # Initializes random weights

# For each epoch & image: feed-forward, backpropagate, update weights
trainBNN(
    network, trainingData, trainingLabels,
    learningRate, epochs, activationFunction
)

let result = classify(network, image)
echo(singleOutcome(result))
```

### Sample w/ MNIST data

Install:

```sh
nimble install https://github.com/marcolussetti/simpleneuralnets
nimble install https://github.com/marcolussetti/mnist_tools
```

Code:

```nim
import simpleneuralnets, mnist_tools

# Load MNIST data
let (trainingLabels, trainingData) = mnistTrainingData()
let (testLabels, testData) = mnistTestData()

# Normalize data
let normTrainingData = trainingData.map(proc(image: seq[int]): seq[
        float] = image / 255.0)
let normTestData = testData.map(proc(image: seq[int]): seq[
        float] = image / 255.0)

# Set hyperparameters
let learningRate = 0.5
let epochs = 20
let activation = tanhActivation
let networkShape = @[784, 20, 20, 10]

# Create new network
var network = newNetwork(networkShape)

# Train network w/ testing data
trainBNN(
    network, normTrainingData, trainingLabels, learningRate,
    epochs, tanhActivation
)

# Classify a single test data point
echo classify(network, normTestData[7])
echo(
    "Expected: ", testLabels[7],
    " Predicted: ", singleOutcome(classify(network, normTestData[7]))
)

# Compute accuracy on training data
# Reverse one-hot encoding
let trainingComputed = lc[singleOutcome(classify(network,
        normTrainingData[
        i])) | (i <- 0..<trainingData.len), int]

let accuracy = computeAccuracy(trainingLabels, trainingComputed)
echo("Accuracy on training data: ", accuracy)

# Test on test data
let testingComputed = lc[singleOutcome(classify(network,
        normTestData[
        i])) | (i <- 0..<testData.len), int]

let testingAccuracy = computeAccuracy(testLabels, testingComputed)
let testingConfMatrix = confusionMatrix(testLabels, testingComputed, 11)

echo("Accuracy on testing data: ", testingAccuracy)
echo("Confusion matrix: ")
echo(testConfMatrix)
```

### More extensive example
For a most extensive use of this library, refer to https://github.com/marcolussetti/simplemnistneuralnetwork
