from sklearn.impute import KNNImputer
from utils import *
import numpy as np
from scipy.spatial.distance import hamming


def pairwise_callable(X, Y, **kwds):
    return hamming(X, Y)


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k, metric=pairwise_callable)
    mat = nbrs.fit_transform(matrix.T).T
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Item Based Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item_weighted(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k, metric=pairwise_callable,
                      weights='distance')
    mat = nbrs.fit_transform(matrix.T).T
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Item Based Weighted Validation Accuracy: {}".format(acc))
    return acc


def main():
    sparse_matrix = load_train_sparse("data").toarray()
    val_data = load_valid_csv("data")
    test_data = load_public_test_csv("data")
    k_list = [1, 6, 11, 16, 21, 26]
    item_accuracies = []
    item_accuracies_weighted = []
    for k in k_list:
        print(f"k = {k}: ")
        item_accuracies.append(knn_impute_by_item(sparse_matrix, val_data, k))
        item_accuracies_weighted.append(
            knn_impute_by_item_weighted(sparse_matrix, val_data, k))
    k_item = int(np.argmax(item_accuracies))
    k_item_weighted = int(np.argmax(item_accuracies_weighted))
    print("\nTest for k*:")
    test_acc_item = knn_impute_by_item(sparse_matrix, test_data, k_list[k_item])
    test_acc_item_weighted = knn_impute_by_item_weighted(sparse_matrix,
                                                         test_data, k_list[
                                                             k_item_weighted])
    print("\n")
    print("Summary")
    print("----------------------")
    print("Item based")
    print("---------------")
    print(f"k selected: {k_list[k_item]}")
    print(f"test acc: {test_acc_item}\n")
    print("Item based weighted")
    print("---------------")
    print(f"k selected: {k_list[k_item_weighted]}")
    print(f"test acc: {test_acc_item_weighted}\n")


if __name__ == "__main__":
    main()