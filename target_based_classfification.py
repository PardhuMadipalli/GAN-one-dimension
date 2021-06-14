import scipy.io as spio
import numpy as np


def normalize_data(dataset):
    return dataset


def target_based_classification(input_mat_file, num_targets):
    dataset = spio.loadmat(input_mat_file)["pxl"]
    original_shape = dataset.shape
    dataset = np.reshape(dataset, (original_shape[0]*original_shape[1], original_shape[2]))
    normalized_data = normalize_data(dataset)
    normalized_data_with_norm = np.linalg.norm(normalized_data, axis=1)
    first_target_location = np.argmax(normalized_data_with_norm)
    # print(first_target_location)
    first_target_value = normalized_data[first_target_location, :]
    num_bands = first_target_value.shape[0]
    U = np.reshape(first_target_value, (num_bands, 1))
    for i in range(1, num_targets):
        middle_value = np.linalg.inv(np.matmul(np.transpose(U), U))
        P_U_perl = np.identity(num_bands) - (np.matmul(np.matmul(U, middle_value), np.transpose(U)))
        y = np.matmul(P_U_perl, np.transpose(normalized_data))
        temp = np.linalg.norm(y, axis=0)
        max_index = np.argmax(temp)
        # print('U shape:', U.shape)
        max_values = np.reshape(np.transpose(normalized_data)[:, max_index], (num_bands, 1))
        # print('max_values shape', max_values.shape)
        U = np.hstack((U, max_values))
        print('max index for target:', i, 'is', max_index)
        # print(U.shape)


target_based_classification("data/HSI_3D.mat", 3)

