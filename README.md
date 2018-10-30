# Autoencoders
Implementing Auto-encoders

If you are not familiar with auto-encoders I recommend to read [this](http://ufldl.stanford.edu/tutorial/unsupervised/Autoencoders/). 

## Simple autoencoder:

The simplest auto-encoder maps an input to itself. This is interesting as the mapping is done by representing the input in a lower dimensional space, that is, compressing the data. Let's implement it.
<br>

```python
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.datasets import mnist

# Load Mnist Data
(trainData,trainLabels),(testData,testLabels)  = mnist.load_data()

```

The sahpe of `trainData` is (60000,28,28), that is, 60K images of 28 by 28 pixels. Now we format the data such that we have new matrices of shape (60000,784). We flattened the image and scale it to have avalues between 0 and 1 by dividing by 255. We do the same with testData, which is of shape (10000,28,28). 
<br>

```python
dim = trainData.shape
trainData = trainData.astype('float32')/255
trainData = trainData.reshape((dim[0],dim[1]*dim[2]))

dim = testData.shape
testData = testData.astype('float32')/255
testData = testData.reshape((dim[0],dim[1]*dim[2]))

```

We then create a model. This model has inputs of 784 elements a single hidden layer of 32 units and the output is 784. This means we will map the 784 pixels to 32 elemets; then we expand the 32 elements to 784 pixels. Lose of information is expected but the amount of compression gained is in most cases worth.
<br>

```python
#%% create the network
model = Sequential()

# Add a dense layer with relu activations and input of 784 elements and 32 units. 
model.add(Dense(32, activation='relu', input_shape=(784,)))

# Connect hidden layer to an output layer with teh same dimension and the input. Sigmoid activations.
model.add(Dense(784, activation='sigmoid')) 

# set the learning parameters
model.compile(optimizer='adadelta', loss='binary_crossentropy')

#learn, use 10 percent for validation (just to see differences between training and testing performance)
model.fit(trainData,trainData,batch_size=256,epochs=50, validation_split = 0.1)
```

We have now learned the network coefficients, let's see how well it reconstruct the inputs using the first five trials as an example. 
<br>
```python
# predict the output
output = model.predict(testData)
output = output.reshape((len(output),28,28))
testData = testData.reshape((len(testData),28,28))

for i in range(0,5):
    ax = plt.subplot(2,5,i + 1)
    plt.imshow(output[i,:,:])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    
    ax = plt.subplot(2,5,i + 1 + 5)
    plt.imshow(testData[i,:,:])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

```

![Figure 1](https://github.com/jfdelgad/Autoencoders/blob/master/output_input.png)

where the first row of images show the output and the second the input. We can see that some information is lost but is possible to distinguish the digits.
<br>

We can take a look at the coefficients (weights) that the models learned. We are interested on the weights that map the input to the hidden layer. We have 32 set of 784 weights. The can be plotted doing:

```python
w = model.get_weights()

for i in range(0,32):
    ax = plt.subplot(4,8,i+1)
    tmp = w[0][:,i].reshape((28,28))
    plt.imshow(tmp)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
```

![Figure 2](https://github.com/jfdelgad/Autoencoders/blob/master/weights.png)


There is one set of coefficients related to ech hidden neuron. Each image then show the pattern in the input that will activate maximally each neuron in the hidden layer.

The main idea is that this method allow to extract the main features needed to representthe data. We could build deeper networks expecting that each layer will make a higher level abstraction compare dto the previous one. Lets see how that work.


## Deep autoencoders:
To do.


