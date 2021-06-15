import scipy.io as spio
import numpy as np


def normalize_data(dataset):
    return dataset


def target_based_classification(input_mat_file, num_targets, top_n=20, threshold_degrees=4,
                                output_name='target_based_samples'):
    print("Running target based sample generation from file", input_mat_file)
    dataset = spio.loadmat(input_mat_file)["pxl"]
    original_shape = dataset.shape
    dataset = np.reshape(dataset, (original_shape[0]*original_shape[1], original_shape[2]))
    normalized_data = normalize_data(dataset)
    normalized_data_with_norm = np.linalg.norm(normalized_data, axis=1)
    first_target_location = np.argmax(normalized_data_with_norm)
    first_target_value = normalized_data[first_target_location, :]
    class_values = find_class_elements(first_target_value, normalized_data_with_norm, normalized_data,
                                       top_n=top_n, threshold_degrees=threshold_degrees)
    num_bands = first_target_value.shape[0]
    final_class_values = np.empty((0, num_bands))
    final_class_count = []

    final_class_values = np.concatenate((final_class_values, class_values))
    final_class_count.append(class_values.shape[0])
    U = np.reshape(first_target_value, (num_bands, 1))
    for i in range(1, num_targets):
        middle_value = np.linalg.inv(np.matmul(np.transpose(U), U))
        P_U_perl = np.identity(num_bands) - (np.matmul(np.matmul(U, middle_value), np.transpose(U)))
        y = np.matmul(P_U_perl, np.transpose(normalized_data))
        temp = np.linalg.norm(y, axis=0)
        max_index = np.argmax(temp)
        # print('U shape:', U.shape)
        max_values = np.reshape(np.transpose(normalized_data)[:, max_index], (num_bands, 1))
        class_values = find_class_elements(max_values, temp, normalized_data, top_n=top_n, threshold_degrees=threshold_degrees)
        final_class_values = np.concatenate((final_class_values, class_values))
        final_class_count.append(class_values.shape[0])
        # print('max_values shape', max_values.shape)
        U = np.hstack((U, max_values))
        #print('max index for target:', i, 'is', max_index)

    output_filename = "data/" + output_name + ".mat"
    print("Saving the output in file:", output_filename)
    spio.savemat(output_filename, {"data": np.transpose(final_class_values)})
    print(final_class_count)
    return output_name, final_class_count


# U = (167, 5)
def find_class_elements(class_vector, normalized_data_with_norm, normalized_data, top_n, threshold_degrees):
    top_n_indices = np.argsort(-normalized_data_with_norm)[:top_n]
    # print(normalized_data_with_norm[top_n_indices])
    top_n_values = normalized_data[top_n_indices]
    top_n_norm_values = normalized_data_with_norm[top_n_indices]
    top_n_values_normalized = np.divide(top_n_values, np.reshape(top_n_norm_values, (top_n, 1)))
    class_norm = np.linalg.norm(class_vector)
    # print('top n values shape', top_n_values.shape)
    cos_values = np.dot(top_n_values_normalized, class_vector/class_norm)
    cos_values = np.clip(cos_values, -1, 1)
    angle_values = np.degrees(np.arccos(cos_values))
    angle_values = np.reshape(angle_values, (top_n,))
    class_indices = np.where(angle_values < threshold_degrees)
    # print('matched:', top_n_values[class_indices].shape)
    return top_n_values[class_indices]


# target_based_classification("data/HSI_3D.mat", 5)

