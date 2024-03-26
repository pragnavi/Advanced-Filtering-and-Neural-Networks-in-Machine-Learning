from sklearn.impute import KNNImputer
from utils import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the mipssing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    matrix_transpose = np.transpose(matrix)
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = np.transpose(nbrs.fit_transform(matrix_transpose))
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("data").toarray()
    val_data = load_valid_csv("data")
    test_data = load_public_test_csv("data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################

    # User-based collaborative filtering
    k_list = [1, 6, 11, 16, 21, 26]
    val_acc_list = []

    for k in k_list:
        acc = knn_impute_by_user(sparse_matrix, val_data, k)
        val_acc_list.append(acc)
        print('k = {}'.format(k))
    plt.plot(k_list, val_acc_list, color='red')
    plt.scatter(k_list, val_acc_list)
    plt.xlim([0,22])
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy vs K using User-Based Collaborative Filtering')
    plt.savefig('ValAcc_vs_K_User_based_Filtering')
    plt.show()

    # Q1b)
    #k* = 11 has highest performance on validation data
    max_arg = np.argmax(val_acc_list)
    k_optimal = k_list[max_arg]
    print("k* = {}".format(k_optimal))

    print()
    nbrs = KNNImputer(n_neighbors=k_optimal)
    mat = nbrs.fit_transform(sparse_matrix)
    acc_k_opt = sparse_matrix_evaluate(test_data, mat)
    print("The final test accuracy with k* ={} is: {}".format(k_optimal, acc_k_opt))
    print()
    

    # Item-based collaborative filtering
    val_acc_list = []

    for k in k_list:
        acc = knn_impute_by_item(sparse_matrix, val_data, k)
        val_acc_list.append(acc)
        print('k = {}'.format(k))
    
    plt.plot(k_list, val_acc_list, color='blue')
    plt.scatter(k_list, val_acc_list)
    plt.xlim([0,22])
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy vs. K using Item-Based Collaborative Filtering')
    plt.savefig('ValAcc_vs_K_Item_based_Filtering')
    plt.show()

    max_arg2 = np.argmax(val_acc_list)
    k_optimal = k_list[max_arg2]
    print("k* = {}".format(k_optimal))

    print()
    nbrs = KNNImputer(n_neighbors=k_optimal)
    mat = nbrs.fit_transform(np.transpose(sparse_matrix))
    acc_k_opt = sparse_matrix_evaluate(test_data, np.transpose(mat))
    print("The final test accuracy with k* ={} is: {}".format(k_optimal, acc_k_opt))
    print()



    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
