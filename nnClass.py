import numpy as np


class NeuralNetwork():
    def __init__(self, numCols):
        np.random.seed(1)
        self.synaptic_weights = 2 * np.random.random((numCols,1)) - 1 #change the 4 to number of columns of input

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        for iteration in range(training_iterations):
            output = self.think(training_inputs)
            error = training_outputs - output
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
            self.synaptic_weights += adjustments
        print("outputs after training")
        print(output)

    def think(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output
    
if __name__ == "__main__":

    neural_network = NeuralNetwork(5)

    print("Random synaptic weights: ")
    print(neural_network.synaptic_weights)

    training_inputs = np.array([[0,0,1,1,1], [1,1,1,0,1], [1,0,1,0,1], [0,1,1,1,1], [0,1,0,1,1], [1,0,1,0,1]])
    print("Training inputs")
    print(training_inputs)

    training_outputs = np.array([[0,1,1,0,1,0]]).T

    neural_network.train(training_inputs, training_outputs, 100000)

    print("Synaptic weights after training: ")
    print(neural_network.synaptic_weights)


    
    A = str(input("Input 1: "))
    B = str(input("Input 2: "))
    C = str(input("Input 3: "))
    D = str(input("Input 4: "))
    E = str(input("Input 5: "))
    

    print("New situation input data ", A, B, C, D, E)

    print("output data")
    print(neural_network.think(np.array([A, B, C, D, E])))