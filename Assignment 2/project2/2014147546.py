import os
import sys
import cv2
import numpy as np

def getX(img_data):
    X = np.zeros(shape=(len(img_data), img_data[0].flatten().shape[0]), dtype=np.int)
    for i in range(0, len(img_data)):
        X[i] = img_data[i].flatten()

    X = X.transpose()

    means = []

    for i in range(0, X.shape[0]):
        pixel_sum = 0
        for j in range(0, X[i].shape[0]):
            pixel_sum += X[i][j]
        means.append(pixel_sum / X[i].shape[0])

    return X, means

def zeroMean(X, means):
    for i in range(0, X.shape[0]):
        for j in range(0, X[i].shape[0]):
            X[i][j] -= means[i]

    return X, means


def getComponent(img_data):

    X, means = getX(img_data)
    X, means = zeroMean(X, means)

    U, S, VT = np.linalg.svd(X, full_matrices=False)
    S_square = np.square(S)

    eigenvalues = []
    eigenvectors = []
    total_sum = np.sum(S_square)

    partial_sum = 0.0
    for i in range(0, S_square.shape[0]):
        partial_sum += S_square[i]
        eigenvectors.append(U.transpose()[i])
        eigenvalues.append(S[i])
        if (partial_sum / total_sum) >= percentage:
            break

    return eigenvalues, eigenvectors, means, U, S, VT

def reconstruction(U, S, VT, means, eigenvalues, num_rows, num_cols):
    new_S = np.zeros(shape=(S.shape[0], S.shape[0]), dtype=np.float)

    for i in range(0, S.shape[0]):
        for j in range(0, S.shape[0]):
            if i == j and i < len(eigenvalues):
                new_S[i][j] = eigenvalues[i]

    X = np.dot(U, np.dot(new_S, VT))

    for i in range(0, X.shape[0]):
        for j in range(0, X[i].shape[0]):
            X[i][j] += means[i]

    X = X.transpose()

    img_data = []
    for i in range(0, X.shape[0]):
        img = X[i].reshape(num_rows, num_cols)
        img_data.append(img)

    return img_data

def getError(original, reconstructed):
    value_sum = 0
    for i in range(0, original.shape[0]):
        for j in range(0, original.shape[1]):
            value_sum += ((original[i][j] - reconstructed[i][j]) * (original[i][j] - reconstructed[i][j]))

    result = float(value_sum) / float(original.shape[0] * original.shape[1])

    return result

def getProjectionResult(img_data, eigenvectors, means_training, num_rows, num_cols):
    X, means = getX(img_data)
    X, means_training = zeroMean(X, means_training)

    sum_mat = np.zeros(shape=(X.shape[0], X.shape[1]), dtype=float)
    for i in range(0, len(eigenvectors)):
        alpha = np.dot(eigenvectors[i].reshape(1, X.shape[0]), X)
        projected = np.dot(eigenvectors[i].reshape(eigenvectors[i].shape[0], 1), alpha)
        sum_mat = np.add(sum_mat, projected)

    for i in range(0, sum_mat.shape[0]):
        for j in range(0, sum_mat.shape[1]):
            sum_mat[i][j] += means_training[i]

    result = []

    sum_mat = sum_mat.transpose()
    for i in range(0, sum_mat.shape[0]):
        result.append(sum_mat[i].reshape(num_rows, num_cols))

    return result

def getDistanceSquare(img1, img2):
    result = 0

    for i in range(0, img1.shape[0]):
        for j in range(0, img1.shape[1]):
            diff = img1[i][j] - img2[i][j]
            result += (diff * diff)

    return result

if __name__ == '__main__':
    STUDENT_CODE = '2014147546'
    FILE_NAME = 'output.txt'

    if not os.path.exists(STUDENT_CODE):
        os.makedirs(STUDENT_CODE)
    f = open(os.path.join(STUDENT_CODE, FILE_NAME), 'w')

    percentage = float(sys.argv[1])

    training_list = os.listdir("faces_training")
    test_list = os.listdir("faces_test")

    training_data = []
    test_data = []

    for name in training_list:
        img = cv2.imread("faces_training/" + name, 0)
        training_data.append(img)

    for name in test_list:
        img = cv2.imread("faces_test/" + name, 0)
        test_data.append(img)

    f.write("##########  STEP 1  ##########\n")

    train_row = training_data[0].shape[0]
    train_col = training_data[0].shape[1]
    eigenvalues_train, eigenvectors_train, means_train, U_train, S_train, VT_train = getComponent(training_data)

    f.write("Input Percentage: {}\n".format(percentage))
    f.write("Selected Dimension: {}\n".format(len(eigenvalues_train)))
    f.write("\n")

    f.write("##########  STEP 2  ##########\n")
    reconstructed_train = reconstruction(U_train, S_train, VT_train, means_train, eigenvalues_train,
                                         train_row, train_col)

    errors = []
    for i in range(0, len(reconstructed_train)):
        error = getError(training_data[i], reconstructed_train[i])
        cv2.imwrite(STUDENT_CODE + "/" + training_list[i], reconstructed_train[i])
        errors.append(error)

    average_error = sum(errors) / len(errors)

    f.write("Reconstruction error\n")
    f.write("average : {}\n".format(format(average_error, '.4f')))

    for i in range(0, len(errors)):
        f.write("{:02d}: {}\n".format(i+1, format(errors[i], '.4f')))

    f.write("\n")

    f.write("##########  STEP 3  ##########\n")
    test_row = test_data[0].shape[0]
    test_col = test_data[0].shape[1]
    reconstructed_test = getProjectionResult(test_data, eigenvectors_train, means_train, test_row, test_col)

    for i in range(0, len(reconstructed_test)):
        min_distance = getDistanceSquare(reconstructed_test[i], reconstructed_train[0])
        index = 0
        for j in range(1, len(reconstructed_train)):
            distance = getDistanceSquare(reconstructed_test[i], reconstructed_train[j])
            if distance < min_distance:
                min_distance = distance
                index = j
        recognized = training_list[index]
        f.write(test_list[i] + " ==> " + recognized + "\n")

    f.close()


