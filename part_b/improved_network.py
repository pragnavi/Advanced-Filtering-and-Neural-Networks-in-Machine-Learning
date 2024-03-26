
from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch
import matplotlib.pyplot as plt

"""Modification to the autoencoder for Part B (adding more layers).
"""

def load_data(base_path="../data"):
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
    #def __init__(self, num_question, code1, code_vect):
    def __init__(self, num_question, k1, k2):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # self.g = nn.Linear(num_question, code1)
        # self.encode1 = nn.Linear(code1, code_vect)
        # self.decode1 = nn.Linear(code_vect, code1)
        # self.h = nn.Linear(code1, num_question)

        self.g = nn.Linear(num_question, k1)
        self.s = nn.Linear(k1, k2)
        self.t = nn.Linear(k2, k1)
        self.h = nn.Linear(k1, num_question)

    def get_weight_norm(self):
        """ Return ||W^1|| + ||W^2||

        :return: float
        """

        # g_w_norm = torch.norm(self.g.weight, 2) 
        # h_w_norm = torch.norm(self.h.weight, 2) 
        # en1_w_norm = torch.norm(self.encode1.weight, 2) 
        # de1_w_norm = torch.norm(self.decode1.weight, 2) 
        # return g_w_norm + h_w_norm + en1_w_norm + de1_w_norm 

        g_w_norm = torch.norm(self.g.weight, 2) 
        s_w_norm = torch.norm(self.s.weight, 2) 
        t_w_norm = torch.norm(self.t.weight, 2) 
        h_w_norm = torch.norm(self.h.weight, 2) 
        return g_w_norm + s_w_norm + t_w_norm + h_w_norm

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
        l1 = self.g(inputs)
        encode = torch.sigmoid(l1)
        l2 = self.s(encode)
        z = torch.sigmoid(l2)
        l3 = self.t(z)
        decode = torch.sigmoid(l3)
        l4 = self.h(decode)
        out = torch.sigmoid(l4)
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch, train_dic):
    """ Train the neural network, where the objective also includes
    a regularizer.
    :param model: Module
    :param lr: float
    :param lamb: float -> for regularization
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function.
    # discuss possibilities: 1: adding the regularization term only once

    # Tell PyTorch you are training the model.
    model.train()
    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]
    train_loss_record = []
    valid_acc_record = []
    valid_loss_record = []
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

            penalty = (lamb/2)*(model.get_weight_norm())
            loss = torch.sum((output - target) ** 2.) + penalty
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)
        valid_loss = evaluate_loss(model,zero_train_data,valid_data)
        train_loss = evaluate_loss(model,zero_train_data,train_dic)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))
        
        #train_loss_record.append(train_loss)
        valid_loss_record.append(valid_loss)
        train_loss_record.append(train_loss)
        valid_acc_record.append(valid_acc)
    return train_loss_record, valid_loss_record, valid_acc_record
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
    return round(correct / float(total), 4)

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

    #code1 = 100
    #code_vect = 10
    #model = AutoEncoder(train_matrix.shape[1], code1, code_vect)
    best_lr = 0.05
    best_num_epoch = 50
    k1 = 100
    k2 = 10
    best_lamb = 0.001
    model = AutoEncoder(train_matrix.shape[1], k1, k2)

    # Set optimization hyperparameters.
    lr = 0.01
    # num_epoch = 200
    num_epoch = 50
    lamb = 0.000
    train_loss_record, valid_loss_record, valid_acc_record=train(model, best_lr, lamb, train_matrix, zero_train_matrix,
          valid_data, best_num_epoch, train_data)
    # train(model, lr, lamb, train_matrix, zero_train_matrix,
    #       valid_data, num_epoch)
    val_acc = evaluate(model,zero_train_matrix, valid_data)
    test_result = evaluate(model, zero_train_matrix, test_data)
    #print("test accuracy: \n" + str(test_result))
    print("validation accuracy = {}, test accuracy = {} ".format(val_acc,test_result))

    #Extended
    plt.title('Training and Validation Loss for each Epoch')
    plt.plot(train_loss_record,color='blue',label='Training Loss')
    plt.plot(valid_loss_record,color='orange',label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Epochs')
    plt.ylim([0.12,0.24])
    plt.grid(True)
    plt.savefig('neural_network_train3.png')
    plt.show()

    plt.title('Validation Accuracy for Each Epoch')
    plt.plot(valid_acc_record,color='green',label='Validation Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.ylim([0.60, 0.70])
    plt.grid(True)
    plt.savefig('neural_network_valid3.png')
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
