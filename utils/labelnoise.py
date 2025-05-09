import numpy as np
import torch
from numpy.testing import assert_array_almost_equal
from utils.set_seed import set_random_seed

def uniform_noise_cp(n_classes, noise_rate):
    P = np.float64(noise_rate) / np.float64(n_classes - 1) * np.ones((n_classes, n_classes))
    np.fill_diagonal(P, (np.float64(1) - np.float64(noise_rate)) * np.ones(n_classes))
    diag_idx = np.arange(n_classes)
    P[diag_idx, diag_idx] = P[diag_idx, diag_idx] + 1.0 - P.sum(0)
    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P

def pair_noise_cp(n_classes, noise_rate):
    P = (1.0 - np.float64(noise_rate)) * np.eye(n_classes)
    for i in range(n_classes):
        P[i, i - 1] = np.float64(noise_rate)
    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P

def random_noise_cp(n_classes, noise_rate):
    P = (1.0 - np.float64(noise_rate)) * np.eye(n_classes)
    for i in range(n_classes):
        tp = np.random.rand(n_classes)
        tp[i] = 0
        tp = (tp / tp.sum()) * noise_rate
        P[i, :] += tp
    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P

def label_dropout(masks, dropout_rate, random_seed):
    new_masks = []
    for mask in masks:
        n_labels = len(mask)
        idx = np.arange(n_labels)
        idx = idx[0: int((1 - dropout_rate) * n_labels)]
        idx.sort()
        new_masks.append(mask[idx])
    return new_masks

def add_label_noise(labels, cp, random_seed):
    assert_array_almost_equal(cp.sum(axis=1), np.ones(cp.shape[1]))
    n_labels = labels.shape[0]
    noisy_labels = labels.copy()
    rs = np.random.RandomState(random_seed)
    for i in range(n_labels):
        label = labels[i]
        flipped = rs.multinomial(1, cp[label, :], 1)[0]
        noisy_label = np.where(flipped == 1)[0]
        noisy_labels[i] = noisy_label
    return noisy_labels

def label_process(labels, n_classes, noise_type='uniform', noise_rate=0, random_seed=5, debug=True):
    set_random_seed(random_seed)
    print(f"random_seed:{random_seed}")
    assert (noise_rate >= 0.) and (noise_rate <= 1.)
    if debug:
        print('----label noise information:------')
    if noise_rate > 0.0:
        if noise_type == 'clean':
            if debug:
                print("Clean data")
            cp = np.eye(n_classes)
        elif noise_type == 'uniform':
            if debug:
                print("Uniform noise")
            cp = uniform_noise_cp(n_classes, noise_rate)
        elif noise_type == 'random':
            if debug:
                print("Random noise")
            cp = random_noise_cp(n_classes, noise_rate)
        elif noise_type == 'pair':
            if debug:
                print("Pair noise")
            cp = pair_noise_cp(n_classes, noise_rate)
        else:
            cp = np.eye(n_classes)
            if debug:
                print("Invalid noise type for a non-zero noise rate: " + noise_type)
    else:
        cp = np.eye(n_classes)

    if noise_rate > 0.0:
        noisy_labels = add_label_noise(labels.cpu().numpy(), cp, random_seed)
        noisy_train_labels = torch.tensor(noisy_labels).to(labels.device)
    else:
        if debug:
            print('Clean data')
        noisy_train_labels = labels.clone()

    actual_noise_rate = (noisy_train_labels.cpu().numpy() != labels.cpu().numpy()).mean()
    modified_mask = np.arange(labels.shape[0])[noisy_train_labels.cpu().numpy() != labels.cpu().numpy()]
    if debug:
        print('#Actual noise rate %.2f ' % actual_noise_rate)

    return noisy_train_labels, modified_mask
