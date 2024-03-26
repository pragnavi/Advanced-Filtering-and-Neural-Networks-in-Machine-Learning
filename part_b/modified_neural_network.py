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
    if torch.cuda.is_available():
        zero_train_matrix = torch.cuda.FloatTensor(zero_train_matrix)
    else:
        zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)
    return zero_train_matrix, train_matrix, train_data, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, l1=800, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.enc_1 = nn.Linear(num_question, l1)
        self.enc_2  = nn.Linear(l1,k)
        self.dec_1  = nn.Linear(k,l1)
        self.dec_2 = nn.Linear(l1, num_question)

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #Encode the inputs and apply sigmoid
        coded = torch.sigmoid(self.enc_1(inputs))
        coded = torch.sigmoid(self.enc_2(coded))
        #Decode the encoded repressentation and apply sigmoid
        decoded = torch.sigmoid(self.dec_1(coded))
        out = torch.sigmoid(self.dec_2(decoded))
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
    # Tell PyTorch you are training the model.
    model.train()
    #Loss lists to return
    valid_acc_list = []
    train_acc_list = []
    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    counter = 0
    for epoch in range(0, num_epoch):
        #print('Epoch {}'.format(epoch))
        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            #Add dropout to the input term
            dropout = torch.nn.Dropout(p=0.25)
            output = model(dropout(inputs))
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.)
            #loss = F.binary_cross_entropy_with_logits(output,target)
            #loss = F.binary_cross_entropy(output, target.detach())
            loss.backward()
            optimizer.step()

        #Set the number of epochs between train/val evaluation
        valid_loss,valid_acc = evaluate(model, zero_train_data, valid_data)
        train_loss,train_acc = evaluate(model,zero_train_data,train_dic)
        print("Epoch: {} \tTraining Acc: {:.6f}\t "
                "Valid Acc: {}".format(epoch, train_acc, valid_acc))
        valid_acc_list.append(valid_acc)
        train_acc_list.append(train_acc)
        counter += 1
    return train_acc_list,valid_acc_list

def evaluate(model, train_data, valid_data):
    """Compute the MSE and Accuracy on the current model and
    given data

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    counter = 0
    correct = 0
    loss = 0
    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)
        guess = output[0][valid_data["question_id"][i]].item()
        target = valid_data["is_correct"][i]
        #Get accuracy
        if round(guess) == target:
            correct += 1
        #Increment loss
        loss += (guess - target)** 2.
        counter += 1
    mse = loss/counter
    acc = correct/counter
    return mse,acc

def main():
    TRAIN = True
    #The best test accuracy of the models 
    HIGH_SCORE = 0.6861416878351679
    zero_train_matrix, train_matrix, train_data, valid_data, test_data = load_data()
    # Set model hyperparameters.
    
    optimal_k=100
    optimal_lr = 0.1
    optimal_epoch = 30
    optimal_lamb=0.1
    optimal_l1 = 800
    '''
    optimal_k=10
    optimal_lr = 0.01
    optimal_epoch = 80
    optimal_lamb=0.001
    optimal_l1 = 800
    '''
    
    # create an AutoEncoder class object
    if torch.cuda.is_available():
        print('Using GPU')
        model = AutoEncoder(train_matrix.shape[1],optimal_l1,optimal_k).cuda()
    else:
        print('Using CPU')
        model = AutoEncoder(train_matrix.shape[1],optimal_l1,optimal_k)
    #train(model, optimal_lr, optimal_lamb, train_matrix, zero_train_matrix,valid_data, optimal_epoch,train_data)
    if TRAIN:
        train_acc,valid_acc = train(model, optimal_lr, optimal_lamb, train_matrix, zero_train_matrix,valid_data, optimal_epoch,train_data)
        _,test_acc = evaluate(model,zero_train_matrix, test_data)
        #Save model if it is better than high score
        if test_acc > HIGH_SCORE:
            print('Saving Best Mdel')
            torch.save(model.state_dict(),'bestmodel.pt')
        print('Test accuracy is: {}'.format(test_acc))
        plt.title('Training Accuracy vs. Validation Accuracy over Epochs')
        plt.plot(train_acc,color='blue',label='Training Accuracy')
        plt.plot(valid_acc,color='orange',label='Validation Accuracy')
        plt.legend(loc='best')
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.show()
    else:
        model.load_state_dict(torch.load('bestmodel.pt'))
        _,test_acc = evaluate(model,zero_train_matrix, test_data)
        print('Test accuracy is: {}'.format(test_acc))
    #Uncomment for Grid-Search
    '''
    acc_list = []
    ks = [10,50,100,200,500]
    lrs = [1e-4,1e-3,1e-2,1e-1,0.5]
    lamb = None
    epoch = 15
    #Gridsearch for K and Learning Rate
    for lr in lrs:
        for k in ks:
            model = AutoEncoder(train_matrix.shape[1],k).cuda()
            valid_acc = train(model, lr, lamb, train_matrix, zero_train_matrix,valid_data, epoch,train_data)
            acc_list.append(valid_acc)
    np.save('acc_list.npy',np.array(acc_list))
    n = len(lrs)
    g = sns.heatmap(np.array(acc_list).reshape(n,n))
    plt.xlabel('Encoding Dimension')
    plt.ylabel('Learning Rate')
    g.set_xticklabels(ks)
    g.set_yticklabels(lrs)
    plt.savefig('lr_ks_gridsearch.png')
    plt.show()
    '''

if __name__ == "__main__":
    main()