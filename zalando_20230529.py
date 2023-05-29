"""
zalando.py
David Nilsson - Prime Fitness Studio AB
2023-05-29
"""



# Import needed libraries

import tensorflow as tf
# from tensorflow import keras

# from tensorflow.compat.v1 import keras
import tensorflow.keras as keras
# Checking the version of TensorFlow
print('TensorFlow version:', tf.__version__)
from tensorflow import keras

#import keras_tuner
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.utils  import to_categorical

# Checking the version of Keras
#print('Keras version:',tensorflow.keras.__version__)

# Helper libraries
import numpy as np
import sklearn
from   sklearn.model_selection import train_test_split

# Matlab plotting
import matplotlib
import matplotlib.pyplot as plt



"""
To easier optimize the hyperparameters the function build_model() could be used.

"""


"""

# Defining a Keras model to search optimized hyper parameters
def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Flatten())
    # Tune the number of layers.
    for i in range(hp.Int("num_layers", 1, 3)):
        model.add(
            layers.Dense(
                # Tune number of units separately.
                units=hp.Int(f"units_{i}", min_value=32, max_value=512, step=32),
                activation=hp.Choice("activation", ["relu", "tanh"]),
            )
        )
    if hp.Boolean("dropout"):
        model.add(layers.Dropout(rate=0.25))
    model.add(layers.Dense(10, activation="softmax"))
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# Initializing a Keras tuner based on random search for the model
tuner = keras_tuner.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=5)

# Starting the search for the optimum hyperparameters for the model
tuner.search(x_train, y_train, epochs=5, validation_data=(x_val, y_val))
best_model = tuner.get_best_models()[0]

build_model(keras_tuner.HyperParameters())
"""



"""
# Test for GPU and determine what GPU we have
import sys
if not tf.config.list_physical_devices('GPU'):
     print("No GPU was detected. CNNs can be very slow without a GPU.")
     IN_COLAB = 'google.colab' in sys.modules
     if IN_COLAB:
         print("Go to Runtime > Change runtime and select a GPU hardware accelerator.")
else:
     !nvidia-smi -L
"""



"""
This code is to load the dataset fashion_mnist
"""

# Get Fashion-MNIST training and test data from Keras database (https://keras.io/datasets/)
(train_images0, train_labels0), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

# Define labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Split the training set into a training and a validation set (20% is validation)
train_images, val_images, train_labels, val_labels = train_test_split(train_images0, train_labels0, test_size=0.20)



"""
Testing that the dataset has ben loaded correctly and what the shapes are of the dataframes
"""

# Print som basic information of data set sizes and data sizes
train_no,x,y = train_images.shape
print('No training images:',train_no, ' with image size:',x,'x',y)
label_no = len(train_labels)
if (label_no != train_no) : 
  print('# labels do not match # training images')

test_no,x,y = test_images.shape
label_no = len(test_labels)
print('No test images:',test_no)
if (label_no != test_no) : 
  print('# labels do not match # test images')

val_no,x,y = val_images.shape
label_no = len(val_labels)
print('No val images:',val_no)
if (label_no != val_no) : 
  print('# labels do not match # val images')

classes = np.unique(train_labels)
num_classes = len(classes)
print('Training labels:', np.unique(train_labels), "; That is,", num_classes,"classes." )



"""
Pre-processing and reshaping the data to be able to work with training a model
"""

# Adding an "empty" color dimension for our data sets
train_images = np.expand_dims(train_images, -1)
val_images = np.expand_dims(val_images, -1)
test_images = np.expand_dims(test_images, -1)

# Normalizing the images.
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5
val_images = (val_images / 255) - 0.5



# As these are images (28x28) it can be interesting to plot some as images
image_index = [42, 789] # "Random" images to print

for index in image_index:
  print( 'Label:', class_names[train_labels[index]])
  plt.figure()
  plt.imshow(np.squeeze(train_images[index], axis=-1))
  plt.gray()
  plt.grid(False)
  plt.show(block=False)
  
  

# Rechaping the input shape of the data
input_shape = test_images[0].shape
print("Input shape", input_shape)

# The Keras model will be the simplest Keras model for NN networks. 
# Working with a sequensial model that can easily be added several layers and
# have a good overview if not too big.
# Going for smaller kernel from the start and increasing at the end to gain more
# abrstraction and capture higher semantic information of the patterns of the data
model = Sequential([

# Add a convolution layer
Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape),

# Add a convolution layer 1
Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape),

# Add a convolution layer 2
Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape),

# Add a convolution layer 3
Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape),

# Add a convolution layer 4
Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape),

# Add a convolution layer 5
Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape),

# Add a convolution layer 6
Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape),

# Add a convolution layer 7
Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape),

# Add a convolution layer 8
Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape),

# Add a convolution layer 9
Conv2D(filters=16, kernel_size=(5, 5), padding='same', activation='relu', input_shape=input_shape),

# Add a convolution layer 10
Conv2D(filters=16, kernel_size=(5, 5), padding='same', activation='relu', input_shape=input_shape),

# Add a convolution layer 11
Conv2D(filters=16, kernel_size=(9, 9), padding='same', activation='relu', input_shape=input_shape),

# Add a convolution layer 12
Conv2D(filters=16, kernel_size=(11, 11), padding='same', activation='relu', input_shape=input_shape),

# Flatten the input to prepare the vector for fully connected layers
Flatten(),

# Add a hidden Dense layer
Dense(units=16, activation='relu'),

# Add a hidden Dense layer1
Dense(units=16, activation='relu'),

# Add a an output layer. The output space is the number of classes
# Softmax makes the output as probablity vector of the different classes
Dense(units=num_classes, activation='softmax')

])

model.summary()



# Compiling the model, as a preparation for training
model.compile(
  optimizer='adam',                    # Tried: adam, sgd
  loss='sparse_categorical_crossentropy',      # sparse_categorical_crossentropy
  metrics=['accuracy']              # categorical_accuracy
)



epochs = 15      ## Number of epoch to run
batch_size = 32  ## Mini batch size



"""
Adding a class for an early stopping if there are to little progress in every epoch
"""
class EarlyStoppingAtMinLoss(keras.callbacks.Callback):
    
    #Stop training when the loss is at its min, i.e. the loss stops decreasing.

    #Arguments:
    #patience: Number of epochs to wait after min has been hit. After this
    #number of no improvement, training stops.
    

    def __init__(self, patience=0):
        super(EarlyStoppingAtMinLoss, self).__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no lBinaryCrossentropyonger minimum.
        self.wait = 10
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("loss")
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))



# Training the model
history = model.fit(
  train_images, to_categorical(train_labels),
  epochs=epochs,
  batch_size=batch_size,
  verbose = 1,
  validation_data=(val_images, to_categorical(val_labels)),
  callbacks=[EarlyStoppingAtMinLoss()],
)



"""
Evaluating the model and plots the performance in terms of accuracy and error
"""
epochrange = range(1, epochs + 1)
train_acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']

train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(epochrange, train_acc, 'bo', label='Training acc')
plt.plot(epochrange, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy (model 1)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(epochrange, train_loss, 'bo', label='Training loss')
plt.plot(epochrange, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss (model 1)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()



# Evaluate the model.
test_loss, test_acc = model.evaluate(test_images,to_categorical(test_labels))
print('Test accuracy: %.3f' % test_acc)



"""
EXERCISE PART 1a
Question: "How many parameters does your model have?" 
Answer: The total parameters in my model is: 201 050, and those are all trainable.
Trainable parameters are weight coefficients to adjust to better connect the
relationship between the the neurons, the neurons themselves and the nodes within 
the neuron net, both the input layers neurons and the hidden layers nodes.
The model becomes overtrained when the error of the validation curve increases unproportional.
Based on the graphs, the number of epochs should be where the curve stops following the training 
error curv.

EXERCISE PART 1b
Question: "What test accuracy do you get?"
Answer: 90% validation accuracy at best

Issues were identified when Keras tuner were intruduced, by not recognizing the imported 
libraries. Loading the modules failed constantly by different approaches and libraries. 
Did not work to use Keras tuner to find a better architecture of the hyperparameters on my 
local machine (Win 10, VS Code, Anaconda and CMD.

Adam optimizer seem to perform well on this Zalando MNIST dataset, and deeper net than approx. 
10 Conv2D and approx. 10 dense layers led to low alpha in the grcategorp method were activated.

around 90% in the validation were achived, and when picking up the best parameter values 
during the training, 91% in validation were achived.

2a
Using the earlystopping class to let the model stop if the training and validation not working 
good enough.
I get at better performance of the model when the model is not overtraining.

2b, 2c
SGT were running onto these gradient issues with deeper layers, and this could probably be 
due to a more averaging effect through the nets layers. This regulation effect could 
be beneficial to generalize better to other dataset.

3
Auto Tune did not work since the model is run locally

Analysis
Since the execution stops all the time in Colab and the Keras Tuder not working properly 
on the local machine, there were hard to find a model that works without flaws.
More extensive search for better hyperparameters would be beneficial.
Due to the initialization based on randomness, the resultsof the performance differs from 
time to time.
"""

    

