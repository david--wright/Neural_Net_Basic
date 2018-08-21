import math
import numpy as np

class NeuralLink:
  def __init__(self, linkedNeuron):
    self.linkedNeuron = linkedNeuron
    self.weight = np.random.normal()
    self.dweight = 0.0

class Neuron:
  learnRate = .003
  p = .03

  def __init__(self, layer):
      self.error = 0.0
      self.gradient = 0.0
      self.output = 0.0
      self.links = []
      if layer is not None:
        for neuron in layer:
          link = NeuralLink(neuron)
          self.links.append(link)
  def sumError(self, err):
    self.error += err
  
  def sigmoid(self, x):
    return 1 / (1 + math.exp(-x * 1.0))

  def dSigmoid(self, x):
    return x * (1.0 - x)      

  def feedForward(self):
    sumOutput = 0
    if not self.links:
      return
    for link in self.links:
      sumOutput += link.linkedNeuron.output * link.weight
    self.output = self.sigmoid(sumOutput)
 
  def backPropagate(self):
    self.gradient = self.error * self.dSigmoid(self.output)
    for link in self.links:
      link.dweight = Neuron.learnRate * (link.linkedNeuron.output * self.gradient) + self.p * link.dweight
      link.weight += link.dweight
      link.linkedNeuron.sumError(link.weight * self.gradient)
    self.error = 0

class NeuralNet:
  def __init__(self, topology):
    self.layers = []
    for neuronCount in topology:
      layer = []
      for i in range(neuronCount):
        if not self.layers:
          layer.append(Neuron(None))
        else:
          layer.append(Neuron(self.layers[-1]))
      layer.append(Neuron(None))
      layer[-1].output = 1
      self.layers.append(layer)
  
  def setInput(self, inputs):
    for i,x in enumerate(inputs):
      self.layers[0][i].output = x
  
  def getError(self, target):
    err=0
    for i,x in enumerate(target):
      e = (x - self.layers[-1][i].output)
      err += e ** 2
      err /= len(target)
      err = math.sqrt(err)
      return err

  def feedForward(self):
    for layer in self.layers[1:]:
      for neuron in layer:
        neuron.feedForward()
  
  def backPropagate(self, target):
    for i,x in enumerate(target):
      self.layers[-1][i].error = x - self.layers[-1][i].output
    for layer in self.layers[::-1]:
      for neuron in layer:
        neuron.backPropagate()
  
  def getOutput(self):
    output = []
    for neuron in self.layers[-1]:
        output.append(neuron.output)
    output.pop() #remove bias neuron
    return output


  def getThresholdOutput(self):
    output = []
    for neuron in self.layers[-1]:
      if neuron.output >= .5:
        output.append(1)
      else:
        output.append(0)
    output.pop() #remove bias neuron
    return output

  def train(self, trainData, iterations, printerr):
    for i in range(iterations):
     for j,x in enumerate(trainData['inputs']):
      err = 0
      self.setInput(x)
      self.feedForward()
      self.backPropagate(trainData['outputs'][j])
      err += self.getError(trainData['outputs'][j])
      if printerr:
        print (i,"- error: ", err)
    return err

def main():
  targetError = 0.01
  topology = [2,3,2]
  net = NeuralNet(topology)
  Neuron.learnRate = 0.09
  Neuron.p = 0.015
  inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
  outputs = [[0, 0], [1, 0], [1, 0], [0, 1]]
  trainData = {'inputs':inputs, 'outputs':outputs}
  err = 1
  while err > targetError:
    err = 0
    err = net.train(trainData, 10000, True)
    print ("Training Run 1 - error: ", err)
 
  while True:
    a = input("a = ")
    b = input("b = ")
    net.setInput([float(a),float(b)])
    net.feedForward()
    print (net.getThresholdOutput())

if __name__ == '__main__':
    main()