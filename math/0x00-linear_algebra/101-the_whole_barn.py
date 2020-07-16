#!/usr/bin/env python3
"""adds two matrices"""


def shape(matrix):
    """ function that calculates
        the shape of a matrix
    """
    shape_matrix = []
    shape_matrix.append(len(matrix))

    while type(matrix[0]) == list:
        matrix = matrix[0]
        shape_matrix.append(len(matrix))

    return shape_matrix


def add_matrices(mat1, mat2):
    """Function thah add two matrices"""
    sum_list = []
    sum_list2 = []
    if (len(shape(mat1)) == 1 and
            len(shape(mat2)) == 1) and shape(mat1) == shape(mat2):
        for i in range(len(mat1)):
            sum_list.append(mat1[i] + mat2[i])
        return sum_list

    elif (len(shape(mat1)) == 2 and
            len(shape(mat2)) == 2) and shape(mat1) == shape(mat2):
        for row in range(len(mat1)):
            sum_list = []
            for col in range(len(mat1[row])):
                sum_list.append(mat1[row][col] + mat2[row][col])
            sum_list2.append(sum_list)
        return sum_list2

    elif (len(shape(mat1)) > 2 and
            len(shape(mat2)) > 2) and shape(mat1) == shape(mat2):
        new_ls = []
        new_ls1 = []
        new_ls2 = []
        new_ls3 = []
        for row in range(len(mat1)):
            new_ls2 = []
            for col in range(len(mat1[row])):
                new_ls1 = []
                for i in range(len(mat1[row][col])):
                    new_ls = []
                    for j in range(len(mat1[row][col][i])):
                        new_ls.append(
                            mat1[row][col][i][j] + mat2[row][col][i][j]
                        )
                    new_ls1.append(new_ls)
                new_ls2.append(new_ls1)
            new_ls3.append(new_ls2)
        return new_ls3
    else:
        return None
