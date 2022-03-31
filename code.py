import pickle
import time
import numpy as np
from PIL import Image


def predict(train_data, test_data, train_labels, test_labels, k_l):
    """
    :param train_data: Data matrix of train points
    :param test_data: Data matrix of test points
    :param train_labels: Train data true labels
    :param test_labels: Test data true labels
    :return: Test error
    """
    predicted_labels = {k: [] for k in k_l}
    u_t_x = np.transpose(train_data)
    u_t_z = np.transpose(test_data)
    for col in u_t_z:
        dist = np.linalg.norm(u_t_x - col, axis=1)
        indexes = np.argsort(dist)
        for k in k_l:
            labels = [train_labels[index] for index in indexes[: k]]
            predicted_labels[k].append(popular(labels))
    for k, v in predicted_labels.items():
        predicted_labels[k] = sum([x != y for x, y in zip(v, test_labels)]) / len(test_labels)
    return predicted_labels


def popular(labels_list):
    """
    Finds the most popular label in a list
    :param labels_list: List of labels
    :return: Most popular label
    """
    return max(set(labels_list), key=labels_list.count)


def get_data(files, train=True):
    if train:
        train_data = []
        train_labels = []
    else:
        test_data = []
        test_labels = []
    for file in files:
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            img = dict[b'data']
            if train:
                for label in dict[b'labels']:
                    train_labels.append(label)
            else:
                for label in dict[b'labels']:
                    test_labels.append(label)
            for i in range(len(img)):
                single_img = np.array(img[i])
                single_img_reshaped = np.transpose(np.reshape(single_img, (3, 32, 32)), (1, 2, 0))
                image = Image.fromarray(single_img_reshaped.astype('uint8'))
                image = image.convert("L")
                arr = np.array(image)
                vector = np.ravel(arr)
                if train:
                    train_data.append(vector)
                else:
                    test_data.append(vector)
    if train:
        return np.transpose(np.array(train_data)), train_labels
    else:
        return np.transpose(np.array(test_data)), test_labels


def run_test(k_l, s_l):
    train_data, train_labels = get_data(files)
    test_data, test_labels = get_data(test, train=False)

    # normalize
    mean_vec = np.mean(train_data, axis=1)
    mean = sum(mean_vec) / len(mean_vec)
    train_data = np.apply_along_axis(lambda a: a - mean, 0, train_data)
    test_data = np.apply_along_axis(lambda a: a - mean, 0, test_data)

    # svd
    u, _, _ = np.linalg.svd(train_data, full_matrices=False, compute_uv=True)

    for s in s_l:
        # with PCA:
        start = time.time()

        # with PCA
        u_s = np.transpose(u[:, :s])

        # projection
        X_proj = u_s @ train_data
        Z_proj = u_s @ test_data

        error = predict(X_proj, Z_proj, train_labels, test_labels, k_l)
        end = time.time()
        for k in k_l:
            print(f'PCA results --> s: {s}, k: {k}, error: {error[k]}')
        print(f'time: {end - start}')

    # without PCA:
    start = time.time()
    u = np.transpose(u)

    # projection
    X_proj = u @ train_data
    Z_proj = u @ test_data

    error = predict(X_proj, Z_proj, train_labels, test_labels, k_l)
    end = time.time()
    for k in k_l:
        print(f'Without PCA results --> k: {k}, error: {error[k]}')
    print(f'time: {end - start}')


if __name__ == '__main__':
    s_l = [10, 20, 50]
    k_l = [2, 7, 15]
    files = [f'data_batch_{i}' for i in range(1, 6)]
    test = ['test_batch']
    run_test(k_l, s_l)
