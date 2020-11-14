from PIL import Image
import numpy as np
import os

#début partie d'adaptation des données lier à l'entrainement
data = np.array([])
data1 = np.array([])
imageList = os.listdir("train_texture")
imageData = np.asarray(Image.open('train_texture/' + imageList[0]))
image1Data = np.asarray(Image.open('train_texture/' + imageList[1]))
for y in range(0, 16):
	for x in range(0, 16):
		data = np.concatenate((data, imageData[y, x, 0:3]))
		data1 = np.concatenate((data1, image1Data[y, x, 0:3]))
data = np.vstack((data, data1))
for z in range(0, len(imageList)-2):
	image1Data = np.asarray(Image.open('train_texture/' + imageList[z+2]))
	data1 = np.array([])
	for y in range(0, 16):
		for x in range(0, 16):
			data1 = np.concatenate((data1, image1Data[y, x, 0:3]))
	data = np.vstack((data, data1))
data = data/255
#fin partie

#début partie d'adaptation des données lier à la prédiction
imageLink = "cobblestone.png"
Prediction = Image.open('train_texture/' + imageLink)
Prediction_data = np.asarray(Prediction)
dataPrediction = np.array([])
for y in range(0, 16):
	for x in range(0, 16):
		dataPrediction = np.concatenate((dataPrediction, Prediction_data[y, x, 0:3]))
dataPrediction = dataPrediction/255
#fin

y = np.array(([0, 0, 0],[0, 0, 1],[0, 1, 0],[0, 1, 1]), dtype=float)

#début classe de réseau neuronal
class Neural_Network(object):
  def __init__(self):
    self.inputSize = 768
    self.outputSize = 3
    self.hiddenSize = 48

    self.W1 = np.random.randn(self.inputSize, self.hiddenSize)
    self.W2 = np.random.randn(self.hiddenSize, self.outputSize)

  def forward(self, X):
		
    self.z = np.dot(X, self.W1)
    self.z2 = self.sigmoid(self.z)
    self.z3 = np.dot(self.z2, self.W2)
    o = self.sigmoid(self.z3)
    return o
    
  def backforward(self, X):
	
    self.a = np.dot(X, self.W2.T)
    self.a2 = self.sigmoid(self.a)
    self.a3 = np.dot(self.a2, self.W1.T)
    o = self.sigmoid(self.a3)
    return o

  def sigmoid(self, s):
    return 1/(1+np.exp(-s))

  def sigmoidPrime(self, s):
    return s * (1 - s)

  def backward(self, X, y, o):

    self.o_error = y - o
    #print(self.o_error)
    self.o_delta = self.o_error*self.sigmoidPrime(o)

    self.z2_error = self.o_delta.dot(self.W2.T)
    self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2)

    self.W1 += X.T.dot(self.z2_delta)
    self.W2 += self.z2.T.dot(self.o_delta)

  def train(self, X, y):
        
    o = self.forward(X)
    self.backward(X, y, o)

  def predict(self):
        
    print("Donnée prédite apres entrainement: ")
    #print("Entrée : \n" + str(dataPrediction))
    print("Sortie : \n" + str(np.matrix.round(self.forward(dataPrediction), 2)))
#fin

NN = Neural_Network()
for i in range(1000000):
    print("#" + i)
    print("Sortie prédite: \n" + str(np.matrix.round(NN.forward(data), 3)))
    print("\n")
    NN.train(data, y)

NN.predict()
w = np.array([[0, 0, 0]])
print(imageLink)
print(os.listdir("train_texture"))
backimage = NN.backforward(w)
backimage_reshape = np.uint8(backimage.reshape(16, 16, 3)*255)
IMG_back = Image.fromarray(backimage_reshape)
IMG_back.save("IMG_back.jpg")