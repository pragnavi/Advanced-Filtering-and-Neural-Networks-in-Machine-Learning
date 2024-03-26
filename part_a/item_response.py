from utils import *

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.
    You may optionally replace the function arguments to receive a matrix.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    num_users = data.shape[0]
    num_questions = data.shape[1]
    theta_i =  theta.reshape((num_users, 1))
    beta_j = beta.reshape((1, num_questions))
    C = data

    Z = theta_i - beta_j
    A = np.multiply(C,Z) - np.log(1 + np.exp(Z))
    A[np.isnan(A)] = 0
    log_lklihood = np.sum(A)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.
    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta
    You may optionally replace the function arguments to receive a matrix.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    num_users = data.shape[0]
    num_questions = data.shape[1]
    theta_i = theta.reshape((num_users, 1))
    beta_j = beta.reshape((1, num_questions))

    Z = theta_i - beta_j
    C = data

    A1 = C - sigmoid(Z)
    A1[np.isnan(A1)] = 0
    dl_dtheta = np.sum(A1, axis = 1)
    print(np.shape(theta))
    theta += lr * dl_dtheta

    A2 = (-1)*C + sigmoid(Z)
    A2[np.isnan(A2)] = 0
    dl_dbeta = np.sum(A2, axis = 0)
    beta += lr * dl_dbeta
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.
    You may optionally replace the function arguments to receive a matrix.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    num_users = data.shape[0]
    num_questions = data.shape[1]
    theta = np.zeros(num_users)
    beta = np.zeros(num_questions)

    val_matrix = np.empty((num_users, num_questions))
    val_matrix.fill(np.nan)
    n = len(val_data['user_id'])
    for i in range(n):
        val_matrix[val_data['user_id'][i], val_data['question_id'][i]] = val_data['is_correct'][i]
   
    train_neg_log_likelihoods = []
    val_neg_log_likelihoods = []
    val_acc_lst = []
    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        train_neg_log_likelihoods.append(neg_lld)
        val_neg_lld = neg_log_likelihood(val_matrix, theta=theta, beta=beta)
        val_neg_log_likelihoods.append(val_neg_lld)

        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)
    
    # TODO: You may change the return values to achieve what you want.
    return theta, beta, train_neg_log_likelihoods, val_neg_log_likelihoods, val_acc_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    print(type(sparse_matrix))
    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
  
    hyperparameters = {'iterations': 100, 'lr': 0.001}
    train_data = sparse_matrix.toarray()
    theta, beta, train_neg_log_likelihoods, val_neg_log_likelihoods, val_acc_lst = irt(train_data, val_data, hyperparameters['lr'], hyperparameters['iterations'])
    # plot the log-likelihood curve

    plt.title('Training Neg-Loglikelihood vs Iterations')
    plt.ylabel('Training Neg-Loglikelihood')
    plt.xlabel('Iterations')
    plt.plot(range(hyperparameters['iterations']), train_neg_log_likelihoods, color='red')
    plt.savefig('training_loglikelihood')
    plt.show()

    plt.title('Validation Neg-Loglikelihood vs Iterations')
    plt.ylabel('Validation Neg-Loglikelihood')
    plt.xlabel('Iterations')
    plt.plot(range(hyperparameters['iterations']), val_neg_log_likelihoods, color='green')
    plt.savefig('validation_neg_loglikelihood')
    plt.show()

    x = range(hyperparameters['iterations'])
    y = train_neg_log_likelihoods
    z = val_neg_log_likelihoods

    plt.plot(x, y, color='r', label='train neg-loglikelihood')
    plt.plot(x, z, color='g', label='val neg-loglikelihood')

    plt.xlabel("Iterations")
    plt.ylabel("Neg-Loglikelihood")
    plt.title("Iterations vs. Neg-Loglikelihood")
    plt.legend()
    plt.savefig('neg_loglikelihood_iterations')
    plt.show()

   

    
    # report the valid and test accuracy
    print("lr: {}".format(hyperparameters['lr']))
    print("iterations:", hyperparameters['iterations'])
    val_acc = evaluate(val_data, theta, beta)
    test_acc = evaluate(test_data, theta, beta)
    print("Final Validation Accuracy: {}".format(val_acc))
    print("Final Test Accuracy: {}".format(test_acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)   
    theta_vals = np.sort(theta)
    beta_vals = [887, 1295, 1199]
    probabilities = []
    for i in beta_vals:
        probabilities.append(sigmoid(theta_vals - beta[i]))

    plt.figure()   
    plt.title('Probability of Correct Response for Question J vs Student Ability θ')   
    plt.ylabel('p(c_ij = 1)')                      
    plt.xlabel('theta θ')

    colors = ['blue', 'red', 'green']
    labels = ['j887', 'j1295', 'j1199']
    n = len(probabilities)
    for i in range(n):
        plt.plot(theta_vals, probabilities[i], color=colors[i], label=labels[i])
    plt.legend(loc="best")
    plt.savefig('probability_vs_theta')
    plt.show()

    #####################################################################
   
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()