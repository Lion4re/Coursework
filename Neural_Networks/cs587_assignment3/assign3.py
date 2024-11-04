# # 3rd assignment of CS-587
# ## Build, Train & Test a Multilayer Neural Networks using PyTorch
# 
# You need to submit:
# - This .ipynb or .py file
# - A report (.pdf or .doc) with the requested tasks.
# - The generated TensorBoard summaries
# 
# ### Goals:
# The aim of this assignment is to get familiar with:
# - Building and training a feed forward neural network using the `PyTorch` framework.
# - Using the SGD method for training to apply automatic differentiation based on PyTorch.
# - Tuning the hyperparameters and modifying the structure of your NN to achieve the highest accuracy.
# - Using the torch `nn` modules, optimizers, DataSets, and DataLoaders
# - Using `Tensorboard` to visualize the graph and results.
# 
# ### Dataset:
# - Digits: 10 class handwritten digits
# - It will automatically be downloaded once you run the provided code using the scikit-learn library.
# - Check for info in the following websites:
# - http://scikit-learn.org/stable/auto_examples/datasets/plot_digits_last_image.html
# - http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html#sklearn.datasets.load_digits
# 
# ### Additional Reading Material:
# - [Tutorial on Datasets & DataLoaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
# - [Tutorial on Building a Neural Network with nn.Module](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)
# - [Tutorial on Optimizing Model Parameters](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
# - [How to use TensorBoard with PyTorch](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html)
# - [Visualizing Models, Data, and Training with TensorBoard](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html)
# - [TensorBoard Documentation for PyTorch](https://pytorch.org/docs/stable/tensorboard.html)

# ## Installing **TensorBoard**
# 
# Don't forget to install the `tensorboard` package, which is necessary for the assignment. To do that, you can simply run:
# ```bash
# conda install conda-forge::tensorboard
# ```
# 
# Alternatively, you can use:
# ```bash
# pip install tensorboard
# ```

import numpy as np
import matplotlib.pyplot as plt
import torch

from digits import DigitsDataset
from torch.utils.data import DataLoader

from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

# ### Let's first load and visualize our dataset

train_set = DigitsDataset(train=True)
test_set = DigitsDataset(train=False)

print("="*10 + " Train Set " + "="*10)
train_set.show_statistics()
print("="*10 + " Test Set " + "="*10)
test_set.show_statistics()

train_set.plot_grid(4, 4)

# how can we access a sample from the dataset, given an index?
sample_index = 33

# we can access the sample using the __getitem__ method
# which returns the x, y pair for the given index
# but, the x is normalized in our case
img1, lbl1 = train_set[sample_index]
# we can get the unnormalized version of the sample
# using the unnormalized_sample method (only in our case)
img2, lbl2 = train_set.unnormalized_sample(sample_index)


plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
img1 = img1.reshape(8, 8)
plt.imshow(img1, cmap=plt.cm.gray_r, interpolation='nearest')
plt.title(f"train[{sample_index}]: {lbl1} (normalized)")
plt.subplot(1, 2, 2)
img2 = img2.reshape(8, 8)
plt.imshow(img2, cmap=plt.cm.gray_r, interpolation='nearest')
plt.title(f"train[{sample_index}]: {lbl1} (original)")
plt.show()

# ## 1) Build a model using PyTorch
# 
# - Using PyTorch, build a simple model (one hidden layer)

def init_weights(shape: tuple):
    # initialize the weights of our model with xavier normal distribution
    return nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(shape)))

def evaluate(dataloader, model):
    # find the accuracy of the model on the given dataloader
    size = len(dataloader.dataset)
    model.eval()
    accuracy = 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            accuracy += (pred.argmax(1) == y).type(torch.float).sum().item()
    accuracy /= size
    return accuracy

# TIME TO BUILD YOUR MODEL, LOSS, AND OPTIMIZER

# hyperparameters
input_size = 64
hid_size = 15
output_size = 10
batch_size = 32
learning_rate = 0.01
num_epochs = 10

class Digits2Layer(nn.Module):
    def __init__(self, input_size, hid_size, output_size=10):
        super(Digits2Layer, self).__init__()
        ############################################
        # TODO: define the parameters of the model #
        ############################################
        self.W_h = pass
        self.b_h = pass
        self.W_o = pass
        self.b_o = pass

    def forward(self, x):
        #################################################
        # TODO: implement the forward pass of the model #
        # you can use 'torch.sigmoid'                   #
        #################################################
        h = pass        # hidden layer
        h_act = pass    # hidden layer after activation
        out = pass      # output layer
        return out

model = Digits2Layer(input_size, hid_size, output_size)

#########################################################
# TODO: define the loss using torch.nn.CrossEntropyLoss #
#  with correct parameters -> check the documentation   #
#########################################################
loss_func = pass

##################################################################
# TODO: set the optimizer to be SGD with the given learning rate #
##################################################################
optimizer = pass

# ### 2) Train your model using SGD algorithm and check the generalization on the test set of your dataset.

# Time to train and evaluate your model
# NOTE: You can launch tensorboard by running the following command:
#       tensorboard --logdir=runs

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# use these lists to keep track of the training loss across iterations
# and train/test accuracy across epochs
losses = []
accs_train = []
accs_test = []

# use tensorboard for visualization
writer = SummaryWriter('runs')
########################################
# TODO: use ΤensorΒoard to visualize   #
# the computational graph of the model #
########################################
pass


for e in range(num_epochs):
    for i, (x, y) in enumerate(train_loader):
        y_pred = model(x)
        ##########################
        # TODO: compute the loss #
        ##########################
        loss = pass
        losses.append(loss.item())

        loss.backward()
        ###############################################
        # TODO: update the weights with the optimizer #
        ###############################################
        pass

        ##########################################################
        # TODO: use TensorBoard to visualize for each iteration: #
        #        - the training loss                             #
        #        - the histogram of the weights W_h              #
        ##########################################################
        pass

    ####################################################################
    # TODO: for each epoch, compute the accuracy on train and test set #
    ####################################################################
    acc_train = pass
    acc_test = pass
    accs_train.append(acc_train)
    accs_test.append(acc_test)
    ##############################################################
    # TODO: use TensorBoard to visualize for each epoch the      #
    # accuracy on the train and the test set (use the same plot) #
    ##############################################################
    pass

    print(f"Epoch {e+1}/{num_epochs}, train accuracy: {acc_train:.3f} test accuracy: {acc_test:.3f}")

writer.close()

#########################################################################
# TODO: Plot two figures, using 'losses', 'accs_train', and 'accs_test' #
# 1. Plot the train loss curve                                          #
# 2. Plot the train and test accuracy curves in the same figure         #
# Compare the results with the ones you get from TensorBoard            #
#########################################################################
plt.figure()
pass
plt.show()

plt.figure()
pass
plt.show()

# ### 3) Try different settings for your model to maximize the accuracy
# 
# Play around with the structure of your NN model and fine-tune its hyperparameters.
# 
# - A. Experiment with different hyperparameters. For example, you can try:
#     - learning rate $= 0.001,\dots,0.1$
#     - batch size $= 8,\dots,128$
#     - size of hidden layers $= 5,\dots,25$
#     - different number of epochs
# - B. Try different activation functions (e.g., `ReLU`, `TanH`).
# - C. Try to add more hidden layers and/or increase their size.
# - D. Add L2 regularization (e.g., with regularization strength $10^{-4}$)
# 
# ### **Bonus:** Extra points (up to $+ 15\%$) will be distributed to the top-performing models based on the accuracy on the test set.

##########################################################
# TODO: MAXimize the accuracy on the given dataset       #
# by trying different settings for your model            #
# (you only need to provide the code for one model)      #
##########################################################
pass


