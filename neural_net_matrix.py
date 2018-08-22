import numpy as np
from . import mnist
from matplotlib import pyplot as plt

def gen_image(arr):
    two_d = (np.reshape(arr, (28, 28)) * 255).astype(np.uint8)
    plt.imshow(two_d, interpolation='nearest')
    return plt

class NeuralNet:
    topology = []
    weights = []
    biases = []
    epochs = 0
    batch_size = 20
    learn_rate = .1
    def __init__(self, topology, batch_size, learn_rate):
        self.topology = topology
        self.weights = [np.random.randn(*w) * 0.1 for w in zip(topology, topology[1:])]
        self.biases = [np.random.randn(b, 1) for b in topology[1:]]
        self.batch_size = batch_size
        self.learn_rate = learn_rate
        self.epochs = 0

    def feed_forward(self, X):
        a = [X]
        for w in self.weights:
            a.append(np.maximum(a[-1].dot(w),0))
        return a

    def gradients(self, a, Y):
        gradients = np.empty_like(self.weights)
        delta = a[-1] - Y
        gradients[-1] = a[-2].T.dot(delta)
        for i in range(len(a)-2, 0, -1):
            delta = (a[i] > 0) * delta.dot(self.weights[i].T)
            gradients[i-1] = a[i-1].T.dot(delta)
        return gradients / len (Y)

    def train(self, num_epochs, trainData, testData):
        trX,trY = trainData['X'], trainData['Y']
        for i in range(num_epochs):
            print (self.epochs + i, self.evaluate(testData))
            for j in range(0, len(trX), self.batch_size):
                X, Y = trX[j:j+self.batch_size], trY[j:j+self.batch_size]
                a = self.feed_forward(X)
                self.weights -= self.learn_rate * self.gradients(a, Y)
            
        self.epochs += num_epochs
    
    def evaluate(self, testData):
        prediction = np.argmax(self.feed_forward(testData['X'])[-1], axis=1)
        return (np.mean(prediction == np.argmax(testData['Y'], axis=1)))
    
    def predict(self, inputs):
        return np.argmax(self.feed_forward(inputs)[-1])

def main():
    trainX, trainY, testX, testY, valX, valY = mnist.load_data()
    trainData = {'X':trainX, 'Y':trainY}
    testData = {'X':testX, 'Y':testY}
    validationData = {'X':valX, 'Y':valY}
    batch_size, learn_rate = 20, 0.1
    num_epochs = int(input("Number of epochs to train: "))
    topology = [784,100,10]
    net = NeuralNet(topology, batch_size, learn_rate)
    net.train(num_epochs,trainData,validationData)
    print ("Final -", net.evaluate(testData))
    while(1):
        testIndex = 1
        try:
            testIndex =input("Select index from test data to display: ")
            testImage = testData['X'][int(testIndex)]
            prediction = net.predict(testImage)
            print ("Prediction -", prediction)
            gen_image(testImage).show()
        except Exception as err:
            if testIndex == 'q':
                break
            elif testIndex == 't':
                num_epochs = int(input("Number of epochs to train: "))
                net.train(num_epochs,trainData,validationData)
            else:
                print (err)
                print ("Please enter an integer less than", len(testData['Y']), ",q to quit, or t to continue training")

    


if __name__ == '__main__':
    main()