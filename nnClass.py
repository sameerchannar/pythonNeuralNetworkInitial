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

    num = int(input("How many factors to consider? "))
    neural_network = NeuralNetwork(num)

    print("Random synaptic weights: ")
    print(neural_network.synaptic_weights)

    training_inputs = np.array([[0.5,0.9,1], #image 1
                                [2,0.7,0.8], #image 2
                                [1,0.6,0.2], #image 3
                                [0,0,0.3], #image 4
                                [1,0.5,0.8], #image 5
                                [1,0.9,0.9]]) #image 6
    print("Training inputs:")
    print(training_inputs)

    training_outputs = np.array([[0.97,0.93,0.76,0.83,0.91,0.92]]).T #result for image 1, 2, 3, 4, 5, 6

    neural_network.train(training_inputs, training_outputs, 20000)

    print("Synaptic weights after training: ")
    print(neural_network.synaptic_weights)

    numUserTests = int(input("How many scenarios to predict? "))
    for test in range(numUserTests):
        print("Test number " + str(test + 1))
        A = str(input("Input 1: "))
        B = str(input("Input 2: "))
        C = str(input("Input 3: "))
        #D = str(input("Input 4: "))
        #E = str(input("Input 5: "))
        

        print("New situation input data: ", A, B, C)

        print("Output:")
        print(neural_network.think(np.array([A, B, C])))