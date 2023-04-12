# Deep_Learning
** David Nilsson - Prime Fitness Studio AB **
Displays the solution for a net to deal with the Zalando MNIST dataset.


Issues were identified when Keras tuner were intruduced, by not recognizing the imported libraries.
Loading the modules failed constantly by different approaches and libraries.
Did not work to use Keras tuner to find a better architecture of the hyperparameters on my local machine 
(Win 10, VS Code, Anaconda and CMD.

Adam optimizer seem to perform well on this Zalando MNIST dataset, and deeper net than approx. 10 Conv2D and approx.
10 dense layers led to low alpha in the gradient descent function, and hence the learning stopped making progress and
the early-stop method were activated.

around 90% in the validation were achived, and when picking up the best parameter values during the training, 91% 
in validation were achived.

SGT were running onto these gradient issues with deeper layers, and this could probably be due to a more averaging effect 
through the nets layers. This should regulation effect could be beneficial to generalize better to other dataset.

