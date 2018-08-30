import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Preparing the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Converting the data into an array with users in rows and movies in columns
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Creating the architecture of the Neural Network
class SAE(nn.Module): # Stacked Autoencoder
    def __init__(self, ):
        super(SAE, self).__init__() # Get inherited methods from the Module class
        # First param is the number of features in the input vector
        # Second param is the number of neurons in the first hidden layer
        self.fc1 = nn.Linear(nb_movies, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nb_movies)
        # Activation function that will activate the neurons when the observation goes into the network
        self.activation = nn.Sigmoid() # Performed better than the rectifier activation function
    # x: input vector
    # Establishes different full connections by applying activation functions to activate the right neurons in the network
    def forward(self, x):
        # Encoding of input vector of features into nodes of hidden layer
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x # Vector of predictive ratings
    # Predicts the user's rating for the chosen movie
    def predict(self, target_user_id, target_movie_id):
        input = Variable(training_set[target_user_id-1]).unsqueeze(0)
        output = sae(input)
        output_numpy = output.data.numpy()
        rating = output_numpy[0,target_movie_id-1]
        return rating
    
sae = SAE()
# Criterion to update the weights - Loss function will be the Mean Squared Error
criterion = nn.MSELoss()
# Apply stochastic gradient descent to update the different weights to reduce the error at each epoch
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)

# Training the SAE
nb_epoch = 200
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    # Count the number of users who rated at least one movie
    s = 0.
    for id_user in range(nb_users):
        # Contains all the ratings of different movies by this user
        input = Variable(training_set[id_user]).unsqueeze(0)
        target = input.clone()
        # Consider users who rated at least 1 movie
        if torch.sum(target.data > 0) > 0:
            output = sae(input)
            # Make sure the gradient is computed with respect to the input and not the target
            target.require_grad = False
            # Nonexistent ratings will not count in the computations of the error
            output[target == 0] = 0
            loss = criterion(output, target)
            # Number of movies over the number of movies with non-zero ratings
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
            # The direction to which the weights will be updated
            loss.backward()
            train_loss += np.sqrt(loss.item() * mean_corrector)
            s += 1.
            # The amount by which the weights will be updated
            optimizer.step()
    print('epoch: ' + str(epoch) + ' loss: ' + str(train_loss/s))

# Testing the SAE
test_loss = 0
s = 0.
for id_user in range(nb_users):
    # Take the input of all the movies that the user already watched
    input = Variable(training_set[id_user]).unsqueeze(0)
    # Compare real ratings to predicted ratings to measure loss
    target = Variable(test_set[id_user]).unsqueeze(0)
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.require_grad = False
        # Only consider non-zero ratings
        output[target == 0] = 0
        loss = criterion(output, target)
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.item()*mean_corrector)
        s += 1.
print('test loss: '+str(test_loss/s))