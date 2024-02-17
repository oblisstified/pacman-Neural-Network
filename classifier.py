# classifier.py
# Lin Li/26-dec-2021
#
# Use the skeleton below for the classifier and insert your code here.

from random import randint,random ,uniform
import numpy as np

class Classifier:
    def __init__(self):
        self.nodesInHiddenLayer = 200
        self.numberOfOutputs = 4
        self.inputWeights = []
        self.hiddenLayerWeights = []
        self.numberOfInputs = 25
    

    def relu(self,x):
        return max(0,x)
    
    def reluDerivative(self, x): 
        return 1 * (x > 0)
    
    def softmax(self, x):
        expX = np.exp(x - np.max(x))
        return expX / np.sum(expX, axis=0)

    def softmaxDerivative(self, x):
        expX = np.exp(x - np.max(x))
        return expX / np.sum(expX, axis=0) * (1 - expX / np.sum(expX, axis=0))
    
    #computes the MSE for all data instances
    def computeError(self,data,target):
        totalError = 0
        for i in range(len(data)):
            totalError+=self.computeErrorForOne(data[i],target[i])          
        return totalError
    
    #computes the MSE for one data instance
    def computeErrorForOne(self, dataInstance, targetInstance):
        _,_,_,a2 = self.forwardPropagation(dataInstance)#gives array of 4 probabilities
        # print(a2,targetInstance)
        targetArray = np.zeros(4)
        targetArray[targetInstance] = 1
        squaredDif = (targetArray-a2)**2
        return np.sum(squaredDif)


    #forward propagation for the nueral network
    def forwardPropagation(self,dataInstance):
        #nodes in hidden layer before activation function
        z1= np.dot(self.inputWeights,dataInstance) 

        #nodes in hidden layer after relu activation function
        a1 = np.array([self.relu(x) for x in z1])

        #nodes in the output layer before activation function
        z2 = np.dot(self.hiddenLayerWeights,a1)

        #output after softmax activation function which returns an array of 4 probabilities
        a2 = self.softmax(z2)

        return z1,a1,z2,a2
    
    #funciton that initiliazes weights randomly and picks the weight with the lowest error (not used)
    def randomBackPropagation(self,data,target):
        error = 10000000000000
        minError = error
        for _ in range(2000):
            self.initializeWeights(data)
            error = self.computeError(data,target)
            if error<minError:
                minError = error
                print(minError)

    #back propagation algorithm that changes the weights so the error converges      
    def backPropagationNew(self, data, target, epochs=5000, alpha=0.001):
        for epoch in range(epochs):
            totalError = 0
            for i in range(len(data)):
                z1,a1,z2,a2 = self.forwardPropagation(data[i])

                #HLW = hidden layer weights
                #ILW = input layer weights

                #array of zeros where the correct target is set to 1
                targetArray = np.zeros(4)
                targetArray[target[i]] = 1

                #gradient calculations for HLW
                derrorda2 = 2*(a2 - targetArray)

                derrordz2 = derrorda2*self.softmaxDerivative(z2)
                derrordHLW = np.outer(derrordz2,a1)
               
                #gradient calculations for ILW
                derrordz1 = np.dot(self.hiddenLayerWeights.T,derrordz2)*self.reluDerivative(z1)
                derrordILW = np.outer(derrordz1,data[i])

                #updating the weights 
                self.hiddenLayerWeights -= alpha * derrordHLW
                self.inputWeights -= alpha * derrordILW

                totalError+=self.computeErrorForOne(data[i], target[i])
            
            #learning rate decay
            alpha*=0.997
           
            print(f"Epoch {epoch}, Total Error: {totalError}")
                
    #randomly initializing weights
    def initializeWeights(self):
        self.inputWeights = np.random.randn(self.nodesInHiddenLayer,self.numberOfInputs)
        self.hiddenLayerWeights = np.random.randn(self.numberOfOutputs,self.nodesInHiddenLayer)

    def reset(self):
        self.initializeWeights()
    
    #fully connected neural network with 25 inputs, 1 hidden layer,variable number of nodes in hidden layer and 4 ouputs
    #training the neural network
    def fit(self, data, target):
        self.initializeWeights()
        self.backPropagationNew(data,target)
        total = 0
        for i in range(len(data)):
            if self.predict(data[i]) == target[i]:
                total+=1
        print(total)

    #predicts the output for a given input
    def predict(self, data, legal=None):
        z1,a1,z2,a2 =  self.forwardPropagation(data)
        return np.argmax(a2)
        
