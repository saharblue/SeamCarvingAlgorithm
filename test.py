import numpy as np


def test_seam_removal():
    np.random.seed(0)  # For reproducibility
    M = np.random.rand(10, 10) # Random values between 0 and 1
    # Example 10x10 backtracking matrix
    backtrack_mat = np.zeros_like(M)
    for i in range(1, 10):
        for j in range(10):
            if j == 0:
                backtrack_mat[i, j] = np.argmin(M[i - 1, j:j + 2])  # Can only move center or right
            elif j == 9:
                backtrack_mat[i, j] = np.argmin(M[i - 1, j - 1:j + 1]) - 1  # Can only move left or center
            else:
                backtrack_mat[i, j] = np.argmin(M[i - 1, j - 1:j + 2]) - 1  # Can move left, center, or right

    print(backtrack_mat)
    print(M)
    for i in range(1):
            seam = []
            height, width = M.shape

            min_col = np.argmin(M[-1])
            seam.append(min_col)

            last_min_col = min_col
            for row in range(height - 2, 0, -1):
                index_in_backtrack_matrix = backtrack_mat[row + 1, last_min_col]
                last_min_col = min_col + index_in_backtrack_matrix
                seam.append(last_min_col)

            seam.append(last_min_col + backtrack_mat[1, last_min_col])
            seam.reverse()
            print(seam)  

test_seam_removal()