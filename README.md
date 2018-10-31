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


## Stacked autoencoders:

We can make autoencoders that are deep, menaing that there is more than one hidden layer. But why?

We know that the autoencoder can be used for unsupervised feature extraction. Above we saw that compressing the image from 748 pixels to 32 degrades the image but the digits are clearly identifiable, therefore we has found that the amount of information in the original image is more or less the same in the compressed images. So auto encoders are good.

Now think about a dense neural network used to classify, assume you have N hidden layers. As the number of layers increases the flexibility of our model increases as well, but the amount of data needed increases and the [vanishing gradient problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem) becomes more important. 

Therfore initialization of the network becomens important. We can model the dense network as series of stacked autoencoders, which will allow us to pre train each layer as an autoencoder and put them together at the end.

Lets code it. Assume a classification problem using MNIST. we will have two hidden layers learned with autoencoders a softwax layer in the output.

Data and labels:
```python
import numpy as np
import keras
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Model
# Load Mnist Data
(trainData,trainLabels),(testData,testLabels)  = mnist.load_data()


# Organize data
dim = trainData.shape
trainData = trainData.astype('float32')/255
trainData = trainData.reshape((dim[0],dim[1]*dim[2]))
dim = testData.shape
testData = testData.astype('float32')/255
testData = testData.reshape((dim[0],dim[1]*dim[2]))
trainLabels = keras.utils.to_categorical(trainLabels, num_classes=10)
testLabels = keras.utils.to_categorical(testLabels, num_classes=10)
```

Now lets create the first autoencoder:

```python
coeff = []

# first autoencoder
autoencoder = Sequential()
autoencoder.add(Dense(128, activation='relu', input_shape=(784,)))
autoencoder.add(Dense(784, activation='sigmoid'))
autoencoder.compile(optimizer='rmsprop', loss='binary_crossentropy')
autoencoder.fit(trainData,trainData,batch_size=256,epochs=50, validation_split = 0.1)


# save the encoding part of teh autoencoder to use at the end as initialization of the complete network
w = autoencoder.get_weights()
coeff.append(w[0])
coeff.append(w[1])

#get the output of the hidden layer to be used as input to the next
encoder = Model(inputs=autoencoder.input,outputs=autoencoder.layers[0].output)
encodedInput = encoder.predict(trainData)
```

Now we repeat this with the next layers, note that `encodedInput` will become the input of the next layer:

```python
#%% Second autoencoder
autoencoder = Sequential()
autoencoder.add(Dense(64, activation='relu', input_shape=(128,)))
autoencoder.add(Dense(128, activation='linear'))
autoencoder.compile(optimizer='rmsprop', loss='mean_squared_error')
autoencoder.fit(encodedInput,encodedInput,batch_size=256,epochs=50, validation_split = 0.1)

# save the encoding part of teh autoencoder to use at the end as initialization of the complete network
w = autoencoder.get_weights()
coeff.append(w[0])
coeff.append(w[1])

#get the output of the hidden layer to be used as input to the next
encoder = Model(inputs=autoencoder.input,outputs=autoencoder.layers[0].output)
encodedInput = encoder.predict(encodedInput)
```

Finally the softwax layer:

```python
#%% softmax
sm = Sequential()
sm.add(Dense(10, activation='softmax', input_shape=(64,)))
sm.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])
sm.fit(encodedInput,trainLabels,batch_size=256,epochs=50, validation_split = 0.1)

# save the encoding part of teh autoencoder to use at the end as initialization of the complete network
w = sm.get_weights()
coeff.append(w[0])
coeff.append(w[1])
```


The saved weights are a good tarting point, we can now fine-tune the complete network, staking all teh autoencoders. Note that weights found in the previous stages are used to nitialize the network.

```python
# Dense layers
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(784,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.layers[0].set_weights(coeff[0:2])
model.layers[1].set_weights(coeff[2:4])
model.layers[2].set_weights(coeff[4:])
# set the learning parameters
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])

#learn, use 10 perecnt for validation (just to see differences between training and testing performance)
model.fit(trainData,trainLabels,batch_size=256,epochs=50, validation_split = 0.1)

score = model.evaluate(testData,testLabels)#0.9745
```

This give us an accuracy in the test set of 97.8% not bad but far from being the state of the art. 

The ide aof autoencoders is great, but having as fundament (as shown here) that the images can be compressed sounds pretty simple. We may explore particular patterns that appear in teh signal. We will need some filters that extract the features and allow us to produce a decomposition of the image in fundamental components.

We can use convolutional neural networks, in our case, convolutional autoencoders.


# Convolutional Autoencoders:

To do.


