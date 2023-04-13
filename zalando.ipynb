"""
** @author David Nilsson - Prime Fitness Studio AB **
** 2023-04-10 - Deep Learning 7,5 credits----------**
"""



# Importing needed libraries
"""
The imported libraries contains classes with methods for different 
computations and calculations, like linear algebra and plotting 
data in windows directly on the screen.
Tensor flow handles data storage into tensors, higher dimentional matrixes.
Keras contains classes for linear algebra for modulating matrixes and
creating and modulating machine learning models.
Numpy is containing classes for mathematical computations.
Matplotlib is containing classes for plotting data in windows on screen.
"""
import tensorflow as tf
from tensorflow import keras
print('TensorFlow version:', tf.__version__)

# from tensorflow import keras
# from tensorflow.keras import layers

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.utils  import to_categorical

# print('Keras version:',tensorflow.keras.__version__)

# Helper libraries
import numpy as np
import sklearn
from   sklearn.model_selection import train_test_split

# Matlab plotting
import matplotlib
import matplotlib.pyplot as plt


"""
No GPU was accessible on my computers, and hence, only CPU were used to
do the computations for training and validating the models.
"""


"""
Importing the data set to work with, from keras databases by open API into dataframes
"""
# Get Fashion-MNIST training and test data from Keras database (https://keras.io/datasets/)
(train_images0, train_labels0), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

# Define a list of the labels of the data
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Split the training set into a training and a validation set (20% is validation)
"""
The train data should be around 60-80% of the total data, and a separate data set 
should be used vor testing, or validation of the model. These data should not be mixed.
Here the data is splitted to 80/20.
"""
train_images, val_images, train_labels, val_labels = train_test_split(train_images0, train_labels0, test_size=0.20)




# Print som basic information of data set sizes and data sizes
"""
Displaying information about the data (zalando MNIST-dataset).
"""
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
Preparing the data (pictures) to be able to pass through the deep learning net.
"""
# Add an "empty" color dimension for our data sets
train_images = np.expand_dims(train_images, -1)
val_images = np.expand_dims(val_images, -1)
test_images = np.expand_dims(test_images, -1)

# Normalize the images.
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5
val_images = (val_images / 255) - 0.5



# As these are images (28x28) it can be interesting to plot some as images
"""
Displaying 2 pictures of the data as the pictures after preparation as low resolution pictures.
"""
image_index = [42, 789] # "Random" images to print

for index in image_index:
  print( 'Label:', class_names[train_labels[index]])
  plt.figure()
  plt.imshow(np.squeeze(train_images[index], axis=-1))
  plt.gray()
  plt.grid(False)
  plt.show(block=False)
  
  
  
# We need to give the input shape (i.e. our image shape) to our model
input_shape = test_images[0].shape
print("Input shape", input_shape)

# The Keras model will be the simplest Keras model for NN networks. 
# It is a single stack of layers connected sequentially.
"""
Were trying out several different variants and number of convolutional and dense 
layers and different sizes of kernels.
Got the best results by having smaller kernels (3, 3) and 10 convolutional layers and 
10 fully connected layers. Regular ReLu activation function were used on every layer 
instead of sigmoid or similar. this reduces the computations and minimizes the risk of
the learning rate drop to zero as the net gets deeper.
Between the convolutional and the fully connected layers there has to be a flattening layer.
The softmax function at the end normalizes the data from the last layer between 0-1.
"""
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
Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape),


# Flatten the input. This prepares the vector for fully connected layers.
Flatten(),

# Add a hidden Dense layer
Dense(units=16, activation='relu'),

# Add a hidden Dense layer1
Dense(units=16, activation='relu'),

# Add a hidden Dense layer2
Dense(units=16, activation='relu'),

# Add a hidden Dense layer3
Dense(units=16, activation='relu'),

# Add a hidden Dense layer4
Dense(units=16, activation='relu'),

# Add a hidden Dense layer5
Dense(units=16, activation='relu'),

# Add a hidden Dense layer6
Dense(units=16, activation='relu'),

# Add a hidden Dense layer7
Dense(units=16, activation='relu'),

# Add a hidden Dense layer8
Dense(units=16, activation='relu'),

# Add a hidden Dense layer9
Dense(units=16, activation='relu'),




# Add a an output layer. The output space is the number of classes
#    Softmax makes the output as probablity vector of the different classes
Dense(units=num_classes, activation='softmax')

])

model.summary()



# Compile the model, as a preparation for training
model.compile(
  optimizer='adam',
  loss='categorical_crossentropy',
  metrics=['categorical_accuracy']
)



epochs = 15      ## Number of epoch to run
batch_size = 32  ## Mini batch size





class EarlyStoppingAtMinLoss(keras.callbacks.Callback):
    """
    Class to stop the training if the training goes to slow.
    Parameter "patience" is the number of epochs to wait until stop, if no progress
    is made in the training results.
    """

    def __init__(self, patience=3):
        super(EarlyStoppingAtMinLoss, self).__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
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



# Train the model.
"""
Starts the model training, and has the ability to abort the training with the
callback, and lets me know if abort is done because of slow progress by the
parameter "verbose".
"""
history = model.fit(
  train_images, to_categorical(train_labels),
  epochs=epochs,
  batch_size=batch_size,
  verbose = 1,
  validation_data=(val_images, to_categorical(val_labels)),
  callbacks=[EarlyStoppingAtMinLoss()],
)


"""
Preparing data and plotting the result from training and validation.
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
"""
Printing out the validation performance of the model.
"""
test_loss, test_acc = model.evaluate(test_images,to_categorical(test_labels))
print('Test accuracy: %.3f' % test_acc)



"""
EXERCISE PART 1a
Question: "How many parameters does your model have?" 
Answer: The total parameters in my model is: 201 050 to approx. 420 000, and those are all trainable.
Trainable parameters are weight coefficients to adjust to better connect the
relationship between the the neurons, the neurons themselves and the nodes within 
the neuron net, both the input layers neurons and the hidden layers nodes.

EXERCISE PART 1b
Question: "What test accuracy do you get?"
Answer: At best 91%, but differs from time to time. Less overtraining after this setup.

EXERCISE PART 2a
Used an early stopper by callback.

EXERCISE PART 2b

EXERCISE PART 2c

EXERCISE PART 3
"""
    
    

