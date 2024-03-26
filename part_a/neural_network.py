from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch

import matplotlib.pyplot as plt
import seaborn as sns

def load_data(base_path="data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)
    train_data = load_train_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, train_data, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """ Return ||W^1|| + ||W^2||.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2)
        h_w_norm = torch.norm(self.h.weight, 2)
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        #Encode the inputs and apply sigmoid
        coded = torch.sigmoid(self.g(inputs))
        #Decode the encoded repressentation and apply sigmoid
        out = torch.sigmoid(self.h(coded))
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch,train_dic):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function. 
    
    # Tell PyTorch you are training the model.
    model.train()
    #Loss lists to return
    valid_loss_list = []
    train_loss_list = []
    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]
            #print(torch.sum((output - target)**2.))

            loss = torch.sum((output - target) ** 2.) + (lamb/2)*(model.get_weight_norm())
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)
        valid_loss = evaluate_loss(model,zero_train_data,valid_data)
        train_loss = evaluate_loss(model,zero_train_data,train_dic)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Cost: {}".format(epoch, train_loss, valid_loss))
        valid_loss_list.append(valid_loss)
        train_loss_list.append(train_loss)
    
    return train_loss_list,valid_loss_list
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)

def evaluate_loss(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.
    Get the validation loss instead of accuracy
    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    counter = 0
    for i, u in enumerate(valid_data["user_id"]):
        #Get the predicted output for the i'th user in the
        #list of user id's on the attempted problem 
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)
        target = valid_data["is_correct"][i]
        guess = output[0][valid_data["question_id"][i]].item()
        counter += 1
        total += (guess - target)** 2.
    return total/counter

def main():
    zero_train_matrix, train_matrix, train_data, valid_data, test_data = load_data()

    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    # Set model hyperparameters.
    ks = [10,50,100,200,500]

    # Set optimization hyperparameters.
    config = {
        'k': ks[0],
        'lr': 0.01,
        'num_epochs': 80,
        'lamb': 0.001  # 0.001 for regularized version
    }
    print(f'Configuration: {config}')
    optimal_k = config['k']
    optimal_lr = config['lr']
    optimal_epoch = config['num_epochs']
    optimal_lamb = config['lamb']
    
    # create an AutoEncoder class object
    model = AutoEncoder(train_matrix.shape[1],optimal_k)
    train_loss,valid_loss = train(model, optimal_lr, optimal_lamb, train_matrix, zero_train_matrix,valid_data, optimal_epoch,train_data)
    test_acc = evaluate(model,zero_train_matrix, test_data)
    print('Test accuracy is: {}'.format(test_acc))
    
    plt.title('Training Loss vs. Validation Loss over Epochs')
    plt.plot(train_loss,color='blue',label='Training Loss')
    plt.plot(valid_loss,color='orange',label='Validation Loss')
    plt.legend(loc='best')
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Epochs')
    plt.savefig('neural_network.png')
    plt.show()
    
    #Uncomment for Grid-Search
    #Consumes processing time
    '''
    acc_list = []
    lrs = [1e-4,1e-3,1e-2,1e-1,0.5]
    #Gridsearch for K and Learning Rate
    for lr in lrs:
        for k in ks:
            model = AutoEncoder(train_matrix.shape[1],k)
            valid_acc = train(model, lr, lamb, train_matrix, zero_train_matrix,valid_data, epoch)
            acc_list.append(valid_acc)

    n = len(lrs)
    g = sns.heatmap(np.array(acc_list).reshape(n,n))
    g.set_xticklabels(ks)
    g.set_yticklabels(lrs)
    plt.savefig('lr_ks_gridsearch.png')
    plt.show()
    '''
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
