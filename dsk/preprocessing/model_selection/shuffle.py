import numpy as np


def shuffle(*args, axis=0):

    # Assert that all arguments are of the same dimension in the specified axis (0 == row, 1 == column)
    # assert all(arg.shape[axis] == args[axis].shape[axis] for arg in args)

    # Create an array of a permutation of the indices for 0 - N of the indices of the original arrays
    permutation = np.random.permutation(np.arange(args[0].shape[axis]))
    shuffled = [[] for _ in enumerate(args)]

    for idx in permutation:
        for new_ind_idx, new_ind in enumerate(shuffled):
            if axis == 0:
                try:
                    new_ind.append(args[new_ind_idx][idx, :])
                except IndexError:
                    new_ind.append(args[new_ind_idx][idx])
            elif axis == 1:
                try:
                    new_ind.append(args[new_ind_idx][:, idx])
                except IndexError:
                    new_ind.append(args[new_ind_idx][idx])

    return [s for s in shuffled]
