from torch import no_grad, stack
from torch.utils.data import DataLoader
from torch.nn import Module
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


"""
Functions you should use.
Please avoid importing any other functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch import optim, tensor, tensordot, ones, matmul
from torch.nn.functional import cross_entropy, relu, mse_loss, softmax
from torch import movedim


import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import no_grad

import torch
from torch import nn
from torch.nn import Module, Parameter
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.autograd import no_grad

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import no_grad

class PerceptronModel(nn.Module):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        """
        super(PerceptronModel, self).__init__()
        
        # Initialize weight as a Parameter with shape (dimensions,)
        self.w = nn.Parameter(torch.ones(1, dimensions))

    def get_weights(self):
        """
        Return the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Args:
            x (Tensor): A tensor of shape (1, dimensions), representing a single data point.

        Returns:
            Tensor: A single number (the score from the perceptron).
        """
        # Calculate the dot product between the weights and input x
        return torch.matmul(x, self.w.t())

    def get_prediction(self, x):
        """
        Calculate the predicted class for a single data point `x`.

        Returns:
            int: 1 or -1 based on the score.
        """
        score = self.run(x)
        prediction = torch.sign(score).item()  # Get the sign of the score

        # Ensure that prediction is either 1 or -1, even if score is 0
        if prediction == 0.0:
            prediction = 1  # You can choose to return 1 or -1 here, I choose 1

        return prediction

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        with no_grad():
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
            all_correct = False  # Flag to track when we have 100% accuracy

            while not all_correct:
                all_correct = True
                for batch in dataloader:
                    x = batch['x']
                    label = batch['label']

                    # Make a prediction
                    prediction = self.get_prediction(x)

                    # Check if the prediction is correct
                    if prediction != label.item():
                        # Misclassified, update weights
                        self.w += (label.item() * x.squeeze())  # Update rule: w += label * x
                        all_correct = False  # There was an error, so continue training

                # If all examples were classified correctly, stop the training loop
                if all_correct:
                    print("Training complete! 100% accuracy achieved.")
                    break




class RegressionModel(nn.Module):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        super(RegressionModel, self).__init__()
        
        # Define the layers of the network
        self.hidden1 = nn.Linear(1, 128) 
        self.hidden2 = nn.Linear(128, 128) 
        self.output = nn.Linear(128, 1) 
        
        self.relu = nn.ReLU()
        
        nn.init.xavier_uniform_(self.hidden1.weight)
        nn.init.xavier_uniform_(self.hidden2.weight)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, x):
        """
        Runs the model for a batch of examples.

        Args:
            x: A tensor of shape (batch_size x 1) representing input values.
        
        Returns:
            A tensor of shape (batch_size x 1) containing predicted y-values.
        """
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        x = self.output(x)
        return x

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples using MSE.

        Args:
            x: A tensor of shape (batch_size x 1) representing input values.
            y: A tensor of shape (batch_size x 1) representing the true y-values.
        
        Returns:
            A scalar tensor representing the MSE loss.
        """
        predictions = self.forward(x)
        loss_fn = nn.MSELoss()
        loss = loss_fn(predictions, y)
        return loss

    def train(self, dataset, epochs=2000, batch_size=64, lr=0.0001):  
        """
        Trains the model.

        Args:
            dataset: A PyTorch Dataset object containing data to be trained on.
            epochs: The number of epochs for training.
            batch_size: The batch size for training.
            lr: The learning rate for the optimizer.
        """
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            running_loss = 0.0
            for batch in dataloader:
                x = batch['x']
                y = batch['label']
                
                optimizer.zero_grad()
                loss = self.get_loss(x, y)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()

            avg_loss = running_loss / len(dataloader)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
  








class DigitClassificationModel(nn.Module):
    """
    A model for handwritten digit classification using the MNIST dataset.
    """

    def __init__(self):
        super().__init__()
        
        input_size = 28 * 28  # Flattened image size (28x28)
        hidden_size = 128     # Size of the hidden layer
        output_size = 10      # Number of classes (digits 0-9)

        # Define the model's layers
        self.fc1 = nn.Linear(input_size, hidden_size)  # First fully connected layer
        self.fc2 = nn.Linear(hidden_size, output_size)  # Output layer (no activation function here)

    def run(self, x):
        """
        Runs the model for a batch of examples.
        
        Inputs:
            x: A tensor of shape (batch_size x 784)
        
        Returns:
            A tensor of shape (batch_size x 10), containing predicted logits (raw scores).
        """
        # Flatten input (28x28 -> 784)
        x = x.view(x.size(0), -1)
        
        # Pass input through the first fully connected layer + ReLU activation
        x = F.relu(self.fc1(x))
        
        # Pass the result through the second fully connected layer (output layer)
        x = self.fc2(x)
        
        # Return the logits (raw scores)
        return x

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples using cross-entropy loss.

        Inputs:
            x: A tensor of shape (batch_size x 784) containing the input images
            y: A tensor of shape (batch_size x 10) containing the one-hot encoded labels
        
        Returns:
            A tensor representing the loss value.
        """
        x = x.requires_grad_(True)
        

        # Get predictions (logits) from the model
        logits = self.run(x)
        
        # Use cross-entropy loss, which combines softmax and negative log-likelihood loss
        # We pass the indices of the one-hot encoded labels, hence `y.argmax(dim=1)`
        loss = F.cross_entropy(logits, y)
        return loss

    
    def train(self, dataset, epochs=5, batch_size=64, learning_rate=0.001):
        """
        Trains the model using the provided dataset.

        Args:
            dataset: The dataset object containing MNIST data
            epochs: The number of epochs for training
            batch_size: The batch size for training
            learning_rate: The learning rate for the optimizer
        """
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            super(DigitClassificationModel, self).train()  # Set the model to training mode
            running_loss = 0.0

            for batch_idx, data in enumerate(train_loader):
                # Since data is a dictionary, access 'x' and 'label' keys
                inputs = data['x']
                targets = data['label']

                # Flatten the input (28x28 -> 784)
                inputs = inputs.view(inputs.size(0), -1)

                # Ensure the target is of type long (for cross-entropy loss)
                targets = targets.to(torch.float32)

                optimizer.zero_grad()

                loss = self.get_loss(inputs, targets)
                loss.backward()

                optimizer.step()

                running_loss += loss.item()

            print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}')









class LanguageIDModel(nn.Module):
    """
    A model for language identification at a single-word granularity.
    """

    def __init__(self):
        super(LanguageIDModel, self).__init__()

        # Define model parameters
        self.num_chars = 47  # Size of the input (number of unique characters)
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]  # List of languages
        self.hidden_size = 128  # Size of the hidden state in the RNN
        self.num_languages = len(self.languages)  # Number of languages to classify (5)

        # RNN layer
        '''self.rnn = nn.RNN(input_size=self.num_chars, 
                          hidden_size=self.hidden_size,
                          num_layers=2,
                          batch_first=False)  # We want to operate on (L x B x num_chars)'''
        
        self.rnn = nn.LSTM(input_size=self.num_chars, 
                   hidden_size=self.hidden_size, 
                   num_layers=3,
                   batch_first=False)


        # Output layer (linear transformation to 5 classes)
        self.fc = nn.Linear(self.hidden_size, self.num_languages)

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Inputs:
            xs: a list of tensors with shape (batch_size x self.num_chars),
                each tensor corresponds to one character in a word.

        Returns:
            A tensor of shape (batch_size x num_languages), containing predicted logits.
        """
        # If xs is a single tensor, convert it to a list
        if isinstance(xs, torch.Tensor):  # If xs is a single tensor
            xs = [xs]

        # Stack the list of tensors into a single tensor of shape (L, batch_size, num_chars)
        xs = torch.stack(xs)  # Shape: (L x batch_size x num_chars)

        # Debugging: print the shape of xs to check
        print("Shape of xs:", xs.shape)  # For debugging, print xs shape

        # Ensure the input has the correct dimensions (L x batch_size x num_chars)
        if xs.dim() == 4:  # If xs is 4D, squeeze unnecessary dimensions
            xs = xs.squeeze(0)  # Remove the extra dimension (if any)
        assert xs.dim() == 3, f"Expected xs to be 3D, but got {xs.dim()}D tensor"

        # Pass through RNN
        rnn_out, _ = self.rnn(xs)  # rnn_out: (L x batch_size x hidden_size)
        
        # We need the output of the final timestep (L-th step)
        final_output = rnn_out[-1, :, :]  # Shape: (batch_size x hidden_size)

        # Pass through the fully connected layer to get logits
        logits = self.fc(final_output)  # Shape: (batch_size x num_languages)
        
        return logits

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            xs: a list of tensors, each of shape (batch_size x self.num_chars),
                corresponding to each character in a word.
            y: a tensor of shape (batch_size x 5) (one-hot encoded labels).

        Returns:
            A scalar loss tensor.
        """
        # Get the logits from the model
        logits = self.run(xs)

        # Compute the cross-entropy loss
        loss = F.cross_entropy(logits, y.argmax(dim=1))  # y is one-hot, so we take the argmax for labels
        
        return loss

    def train(self, dataset, epochs=29, batch_size=64, learning_rate=0.001):
        """
        Trains the model using the provided dataset.

        Args:
            dataset: The dataset object containing the training data.
            epochs: The number of epochs to train for.
            batch_size: The batch size for training.
            learning_rate: The learning rate for the optimizer.
        """
        # DataLoader setup for the training dataset
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(epochs):
            super(LanguageIDModel, self).train()  # Set the model to training mode
            running_loss = 0.0

            for batch_idx, data in enumerate(train_loader):
                # Inputs is of shape (batch_size x length_of_word x num_chars)
                # Targets is of shape (batch_size x 5) (one-hot encoded labels)
                inputs = data['x']
                targets = data['label']
                # Switch dimensions from (batch_size x L x num_chars) to (L x batch_size x num_chars)
                inputs = inputs.permute(1, 0, 2)  # Shape: (L x batch_size x num_chars)

                optimizer.zero_grad()

                loss = self.get_loss(inputs, targets)
                loss.backward()

                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")



        


def Convolve(input: torch.Tensor, weight: torch.Tensor):
    """
    Acts as a convolution layer by applying a 2d convolution with the given inputs and weights.
    The convolution must be done manually without directly using PyTorch's conv2d method.
    """
    # Get the dimensions of input and weight
    input_height, input_width = input.shape
    weight_height, weight_width = weight.shape
    
    # Calculate the dimensions of the output tensor
    output_height = input_height - weight_height + 1
    output_width = input_width - weight_width + 1
    
    # Initialize the output tensor with zeros
    Output_Tensor = torch.zeros((output_height, output_width), device=input.device)
    
    # Perform the convolution
    for y in range(output_height):
        for x in range(output_width):
            # Extract the current region of the input matrix
            region = input[y:y+weight_height, x:x+weight_width]
            # Perform element-wise multiplication and sum the result
            Output_Tensor[y, x] = torch.sum(region * weight)
    
    return Output_Tensor



'''
class DigitConvolutionalModel(nn.Module):
    """
    A model for handwritten digit classification using the MNIST dataset.
    """

    def __init__(self):
        super(DigitConvolutionalModel, self).__init__()
        
        # Initialize convolution weights (3x3 kernel)
        self.convolution_weights = Parameter(torch.ones((3, 3)))
        
        # Define the fully connected layers
        self.fc1 = nn.Linear(26 * 26, 128)  # Flattened output of convolution
        self.fc2 = nn.Linear(128, 10)  # 10 classes for digits 0-9

    def forward(self, x):
        """
        The convolutional layer is already applied, and the output is flattened for you. 
        You should treat x as a regular 1-dimensional datapoint now, similar to the previous questions.
        """
        x = x.reshape(len(x), 28, 28)  # Reshape input to (batch_size, 28, 28)
        



        # Apply convolution explicitly without using map
        convolved = []
        for sample in x:
            convolved.append(Convolve(sample, self.convolution_weights))
        x = torch.stack(convolved)
        # Apply convolution
        #x = torch.stack(list(map(lambda sample: Convolve(sample, self.convolution_weights), x)))
        
        # Flatten the output from convolution
        x = x.flatten(start_dim=1)  # Flatten the (batch_size, height, width) to (batch_size, -1)
        
        # Pass through the fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        #x = x.requires_grad_(True)

        return x

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples using cross-entropy loss.
        
        Inputs:
            x: A tensor of shape (batch_size x 784)
            y: A tensor of shape (batch_size x 10)
        Returns: A loss tensor
        """
        if y.dim() > 1:
            y = y.argmax(dim=1)

        x = x.requires_grad_(True)
        return torch.nn.functional.cross_entropy(x, y)

    def train(self, dataset):
        """
        Trains the model using the provided dataset.
        """
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        
        for epoch in range(5):  # Train for 5 epochs
            self.train()  # Set the model to training mode
            running_loss = 0.0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.float()  # Ensure inputs are floats
                targets = targets.long()  # Ensure targets are of type long
                
                optimizer.zero_grad()  # Zero the gradients
                
                # Forward pass
                outputs = self(inputs)
                
                # Compute loss
                loss = self.get_loss(outputs, targets)
                
                # Backward pass
                loss.backward()
                optimizer.step()  # Update the weights
                
                running_loss += loss.item()
            
            print(f'Epoch [{epoch+1}/5], Loss: {running_loss / len(train_loader):.4f}')
'''


class AutomatedDataset(Dataset):
    def __init__(self, x, labels):
        self.x = x
        self.labels = labels

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        entity = self.x[idx] 
        id = self.labels[idx]  
        return entity, id

class DigitConvolutionalModel(nn.Module):
    def __init__(self):
        super(DigitConvolutionalModel, self).__init__()
        
        self.convolution_weights = Parameter(torch.randn(1, 1, 3, 3))  # Filter of size 3x3 for a single channel
        
        # First layer
        self.layer1 = nn.Linear(26 * 26, 256)
        self.relu1 = nn.ReLU()

        # Hidden layer
        self.layer2 = nn.Linear(256, 32)
        self.relu2 = nn.ReLU()

        # Output layer: digits 0-9 to 10 classes
        self.layer3 = nn.Linear(32, 10)

    def run(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        The convolutional layer is already applied, and the output is flattened for you.
        You should treat x as a regular 1-dimensional datapoint now, similar to the previous questions.
        """
        x = x.reshape(len(x), 28, 28) # Reshape input to (batch_size, 28, 28)

        
        result = []
        for sample in x:
            convolved_sample = self.Convolve(sample, self.convolution_weights)
            result.append(convolved_sample)
        x = torch.stack(result)


        x = x.flatten(start_dim=1)
    

        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.layer3(x) 

        return x

    def Convolve(self, input: torch.Tensor, weight: torch.Tensor):
        """
        Applies a 2D convolution on the input tensor using the given weights.
        """
        input_tensor = input.unsqueeze(0).unsqueeze(0)
        
        output_tensor = F.conv2d(input_tensor, weight)

        return output_tensor

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.
        The correct labels `y` are represented as a tensor with shape (batch_size x 10).
        Each row is a one-hot vector encoding the correct digit class (0-9).
        """
        predictions = self.forward(x)  
        return torch.nn.functional.cross_entropy(predictions, y)
    

    def train(self, dataset):
        """
        Trains the model.
        """
        optimizer = optim.SGD(self.parameters(), lr=0.1)
        epochs = 60
        batch_size = 128
        train_data_x = dataset.x
        train_data_y = dataset.y
        train_dataset = AutomatedDataset(train_data_x, train_data_y)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            for batch in train_loader:
                x = batch[0]
                y = batch[1]

                optimizer.zero_grad()
                loss = self.get_loss(x, y)
                loss.backward()
                optimizer.step()






class Attention(Module):
    def __init__(self, layer_size, block_size):
        super().__init__()
        """
        All the layers you should use are defined here.
        """

        # Initialize layers generating Q, K, V matrices
        self.q_layer = Linear(layer_size, layer_size)
        self.k_layer = Linear(layer_size, layer_size)
        self.v_layer = Linear(layer_size, layer_size)

        # Masking 
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))
        self.layer_size = layer_size

    def forward(self, input):
        """
        Applies the attention mechanism to input. All necessary layers have been defined in __init__().
        """
        B, T, C = input.size()

        Q = self.q_layer(input)
        K = self.k_layer(input)
        V = self.v_layer(input)

        scores = torch.matmul(K, Q.transpose(-1, -2)) / (self.layer_size ** 0.5)
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)

        return output
     