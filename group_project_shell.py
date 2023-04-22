import numpy as np
class BasicLinAlg:
    """
    This class is a collection of static methods for performing various operations on arrays and matrices.

    This class provides methods for performing basic arithmetic operations (addition, subtraction, 
    scalar multiplication, dot product, vector norm) on vectors, and matrix operations such as 
    matrix multiplication, transposition, determinant, inverse, matrix addition and subtraction, 
    matrix power, rank, shape, reduced row echelon form, LU decomposition, QR decomposition,
    eigenvalues and eigenvectors.

    All the methods are implemented as static methods and do not require an instance of the class to be created. 
    The input arrays and matrices are expected to be represented as Python lists, with each element of the list 
    representing a row or column of the array/matrix.

    Numpy is used only to calculate the characteristic polynomial and its roots for the computation of 
    eigenvalues. 

    Some methods, such as eigenvectors and check_eigenvalues, utilize an epsilon (eps) value to compare 
    floating-point numbers with a tolerance level. The default eps value is set to 1e-10, which can be changed
    if needed. This tolerance level helps avoid issues related to floating-point inaccuracies, especially when
    comparing values that are close to zero.
    """
    
    eps = 1e-10 
    @staticmethod
    def scalar_multiply_vector(vector, scalar):
        """
        Method -- scalar_multiply_vector
        This method returns the result of multiplying a vector by a scalar.

        Parameters:
        vector (list): The vector to multiply.
        scalar (float or int): The scalar to multiply the vector by.

        Returns:
        list: A list representing the result of multiplying the input vector by the input scalar.
        """
        # set row to false assuming vector passed in as column
        row = False
        if len(vector) == 1: # transpose to column
            vector = BasicLinAlg.transpose(vector)
            row = True # indicates transpose happened
        
        # Error handling
        elif len(vector) == 0: 
            raise ValueError("Empty vector")
        
        elif len(vector) > 1:
            for i in range(len(vector)):
                if len(vector[i]) != 1:
                    raise ValueError("Input not a vector")
        # initialize new vector    
        new_vector = []
        for i in range(len(vector)):
            new_vector.append([]) # add column entries
        
        for i in range(len(vector)):
            new_vector[i].append(vector[i][0]*scalar) # multiply each entry by scalar and append

        if row == True: # check to see if need to transpose to original form
            new_vector = BasicLinAlg.transpose(new_vector)
    
        return new_vector
    
    @staticmethod
    def vector_norm(vector):
        """
        Method -- vector_norm
        This method computes and returns the Euclidean norm of a vector, which can be represented as a row or a column vector.

        Parameters:
        vector (list): The vector to find the norm of, represented as a nested list. 
                       For a row vector, use [[1, 2, 3]]; 
                       For a column vector, use [[1], [2], [3]].

        Returns:
        float: The Euclidean norm of the input vector.
        """
        # Check for empty vector
        if not vector or not vector[0]:  
            return 0
        
        # Column vector
        if len(vector[0]) == 1:  
            return sum(x[0]**2 for x in vector) ** 0.5
        
        # Row vector
        else:  
            result = sum(x**2 for x in vector[0]) ** 0.5
            return result
    
    @staticmethod
    def dot_product(v1, v2):
        """
        Method -- dot_product
        This method computes and returns the dot product of two vectors of the same length.

        Parameters:
        v1 (nested list): The first vector to take the dot product of.
        v2 (list): The second vector to take the dot product of.

        Returns:
        float: The dot product of the two input vectors.
        """
        
        # row vectors
        if len(v1) == len(v2) and len(v1) == 1:
            v1 = BasicLinAlg.transpose(v1)
            v2 = BasicLinAlg.transpose(v2)

        # mixed vector convention handling
        elif len(v1) != len(v2):

            # change v1 to column vector
            if len(v1) == 1:
                 v1 = BasicLinAlg.transpose(v1)

            #change v2 to column vector
            if len(v2) == 1:
                 v2 = BasicLinAlg.transpose(v2)

        if len(v1) != len(v2):
            raise ValueError('Vectors must be the same length')

        return sum(v1[i][j] * v2[i][j] for i in range(len(v2)) for j in range(len(v2[i])))
            
    @staticmethod
    def transpose(m1):
        """
        Method -- transpose 
        This method computes and returns the transpose of a matrix.

        Parameters:
        matrix (list of lists): The matrix to find the transpose of.

        Returns:
        list of lists: A list of lists representing the transpose of the input matrix.
        """
        if not m1 or not m1[0]:  # Check for empty matrix or matrix with an empty row
            return []
        
        trans = []
        for i in m1[0]:
            trans.append([])

        for i in range(len(m1[0])):
            for j in range(len(m1)):
                trans[i].append(m1[j][i])
            
        return trans
    
    @staticmethod
    def determinant(matrix):
        """
        Method -- determinant
        This method calculates the determinant of a square matrix.

        Parameters:
        matrix (list of lists) -- The input square matrix.

        Returns:
        det (float) -- The determinant of the input matrix.

        Raises:
        ValueError -- If the input matrix is not square.
        """
        rows = len(matrix)
        
        if rows == 0:  # Added this check for empty matrix
            return 1

        cols = len(matrix[0])

        if rows != cols:
            raise ValueError("Matrix must be square")
        
        if rows == 1:
            return matrix[0][0]
        
        if rows == 2:
            return (matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0])

        zeros = [row.count(0) for row in matrix]
        opti = zeros.index(max(zeros))  # row with most zeros = most optimal
        detsum = 0
        for i in range(cols):
            submatrix = BasicLinAlg.get_submatrix(matrix, row=(opti + 1), col=(i + 1))
            detsum += matrix[opti][i] * (-1) ** (1 + (i + 1)) * BasicLinAlg.determinant(submatrix)
        
        return detsum

    @staticmethod
    def get_submatrix(matrix, row = None, col = None):
        """
        Method -- _get_submatrix
        This method is a helper method to get the submatrix formed by deleting the given row and column from the input matrix.

        Parameters:
        matrix (list of lists) -- The input matrix.
        row (int) -- The row to delete.
        col (int) -- The column to delete.

        Returns:
        submatrix (list of lists) -- The submatrix formed by deleting the
        given row and column from the input matrix.
        """
        # had to google. static instance was messing up my matrix[:] copy
        sub = [row[:] for row in matrix]  

        # knock off a row
        if row != None:
            del sub[row - 1]        

        # knock off a col
        if col != None:
            for i in range(len(sub)):
                del sub[i][col-1]
                
        return sub

    @staticmethod
    def inverse(matrix):
        """
        Method -- inverse
        Returns the inverse of a square matrix.

        Parameters:
        matrix (list of lists): The matrix to find the inverse of.

        Returns:
        list of lists: A list of lists representing the inverse of the input matrix.

        Raises:
        ValueError: If the input matrix is empty, not square, or singular.
        """
        ad_matrix = BasicLinAlg.adjoint(matrix) # get adjoint matrix
        deter = BasicLinAlg.determinant(matrix) # get determinant

        if deter == 0: # check that matrix has inverse
            raise ValueError("Matrix is singular")
        
        inv = BasicLinAlg.scalar_multiply(ad_matrix, (1/deter))
        return inv 


    @staticmethod
    def adjoint(matrix):
        """
        Method -- adjoint
        This method computes and returns the adjoint of a square matrix.
        The adjoint of a matrix is the transpose of the cofactor matrix.
        
        Parameters:
        matrix (list of lists): The input square matrix to find the adjoint of.

        Returns:
        list of lists: A list of lists representing the adjoint of the input matrix.

        Raises:
        ValueError: If the input matrix is not square.
        ValueError: If the input matrix is empty.
        """
        if not matrix:  # Check for empty matrix
            raise ValueError("Matrix must be non-empty")
        
        rows = len(matrix)
        cols = len(matrix[0])

        if rows != cols:
            raise ValueError("Matrix must be square")

        adj = [[0 for _ in range(cols)] for _ in range(rows)]

        for i in range(rows):
            for j in range(cols):
                submatrix = BasicLinAlg.get_submatrix(matrix, row=(i + 1), col=(j + 1))
                cofactor = ((-1) ** (i + j)) * BasicLinAlg.determinant(submatrix)  # Reverted sign calculation
                adj[j][i] = cofactor

        return adj

    @staticmethod
    def scalar_multiply(matrix, scalar):
        """
        Method -- scalar_multiply
        This method performs scalar multiplication of a matrix.

        Parameters:
        matrix (list of list of float/int) -- The input matrix to be multiplied.
        scalar (float/int) -- The scalar value to multiply the matrix.

        Returns:
        list of list of float/int -- The scalar product of the matrix.
        """

        return [[scalar * entry for entry in row] for row in matrix]

    @staticmethod
    def matrix_addition(m1, m2):
        """
        Method -- matrix_addition
        This method performs matrix addition of two matrices.

        Parameters:
        m1 (list of list of float/int) -- The first input matrix to be added.
        m2 (list of list of float/int) -- The second input matrix to be added.

        Returns:
        list of list of float/int -- The sum of the two input matrices.
        """
        
        # if the matrices are not the same size
        if BasicLinAlg.shape(m1) != BasicLinAlg.shape(m2):
            raise ValueError("The Matrices are not the same size")
        
        else:
            m3 = []
            for i in range(len(m1)): # per row
                row = []
                for j in range(len(m1[0])): # loop through the columns
                    row.append(m1[i][j] + m2[i][j])
                m3.append(row)        
            return m3
                                
    @staticmethod
    def matrix_subtraction(m1, m2):
        """
        Method -- matrix_subtraction
        This method performs matrix subtraction of two matrices.

        Parameters:
        m1 (list of list of float/int) -- The first input matrix to be subtracted.
        m2 (list of list of float/int) -- The second input matrix to be subtracted.

        Returns:
        list of list of float/int -- The difference of the two input matrices.
        """
        # if the matrices are not the same size
        if BasicLinAlg.shape(m1) != BasicLinAlg.shape(m2):
            raise ValueError("The Matrices are not the same size")
        else:
            m3 = []
            for i in range(len(m1)): # per row
                row = []
                for j in range(len(m1[0])): # loop through the columns
                    row.append(m1[i][j] - m2[i][j])
                m3.append(row)
                
            return m3

    @staticmethod
    def matrix_multiply(m1, m2):
        """
        Method -- matrix_multiply
        This funciton returns the product of two matrices.

        Parameters:
        m1 (list of lists): The first matrix to multiply.
        m2 (list of lists): The second matrix to multiply.

        Returns:
        list of lists: A list of lists representing the product of the two input matrices.
        """
        # if columns of 1 != rows of 2
        if BasicLinAlg.shape(m1)[1] != BasicLinAlg.shape(m2)[0]:
            raise ValueError("The Matrices cannot be multiplied, check dimensions")
            
        else:
            # zero matrix w proper dimensions 
            m3 = [[0 for j in range(len(m2[0]))] for i in range(len(m1))] 


            # Flip the matrix - (Remove when we have transpose function since verbose)
            m2tpose = BasicLinAlg.transpose(m2)


            # Dot product for flipped matrix.
            for i in range(len(m3)):
                for j in range(len(m3[0])):
                    m3[i][j] = BasicLinAlg.dot_product([m1[i]],[m2tpose[j]])

            return m3
    
    @staticmethod
    def rref(matrix):
        """
        Given a matrix, return its reduced row echelon form.

        Parameters:
        matrix (nested list): The matrix for which to find the reduced row echelon form.

        Returns:
        nested list: The reduced row echelon form of the given matrix.
        """
        lead = 0
        rowCount = len(matrix)
        columnCount = len(matrix[0])
        for r in range(rowCount):
            if lead >= columnCount:
                break
            i = r
            while abs(matrix[i][lead]) <= BasicLinAlg.eps:
                i += 1
                if i == rowCount:
                    i = r
                    lead += 1
                    if columnCount == lead:
                        break

            if columnCount != lead:
                matrix[i], matrix[r] = matrix[r], matrix[i]
                if matrix[r][lead] != 0:
                    lv = matrix[r][lead]
                    matrix[r] = [mrx / lv for mrx in matrix[r]]

                for i in range(r + 1, rowCount):
                    if matrix[i][lead] != 0:
                        lv = matrix[i][lead]
                        matrix[i] = [iv - lv * rv for rv, iv in zip(matrix[r], matrix[i])]

                lead += 1

        for r in range(rowCount - 1, 0, -1):
            lead = -1
            for c, elem in enumerate(matrix[r]):
                if abs(elem) > BasicLinAlg.eps:
                    lead = c
                    break

            if lead != -1:
                for i in range(r):
                    if matrix[i][lead] != 0:
                        lv = matrix[i][lead]
                        matrix[i] = [iv - lv * rv for rv, iv in zip(matrix[r], matrix[i])]

        return matrix

    
    @staticmethod
    def shape(matrix):
        """
        Given a matrix, return its shape as a tuple (rows, cols).

        Parameters:
        matrix (list of list): The matrix for which to find the shape.

        Returns:
        tuple: A tuple representing the shape of the matrix as (rows, cols).
        """
        if len(matrix) > 0: 
            return(len(matrix),len(matrix[0]))
        return (len(matrix),0)

    @staticmethod
    def rank(matrix):
        """
        Given a matrix, return its rank.

        Parameters:
        matrix (list): The matrix for which to find the rank.

        Returns:
        int: The rank of the given matrix.
        """
        rref_matrix = BasicLinAlg.rref(matrix)
        rank = sum([1 for row in rref_matrix if any(elem != 0 for elem in row)])
        return rank
        
    @staticmethod
    def matrix_power(matrix, power):
        """
        Method -- matrix_power
        This method computes the power of a matrix.

        Parameters:
        matrix (list of list of float/int) -- The input matrix to be powered.
        power (int) -- The power to which the matrix is to be raised.

        Returns:
        list of list of float/int -- The matrix raised to the given power.
        """
        if power <= 0 or type(power) != int:
            raise ValueError('Power must be an integer greater than 0')
        if BasicLinAlg.shape(matrix)[0] != BasicLinAlg.shape(matrix)[1]:
            raise ValueError("Matrix must be square, check dimensions")

        matrix_power = [each for each in matrix]
        # loop power times
        for times in range(1,power):
            # multiply matrix time itself
            matrix_power = BasicLinAlg.matrix_multiply(matrix, matrix_power)
        return matrix_power

    @staticmethod
    def identity(size):
        """
        Method -- identity
        This method creates an identity matrix of given size.

        Parameters:
        size (int) -- The size of the identity matrix to be created.

        Returns:
        list of list of float/int -- The identity matrix of the given size.
        """
        # make empty matrix to store identity
        id_matrix = []
        # special case
        if size == 0:
            return [[]]
        else:
            # make matrix entirely of 0
            for i in range(size):
                id_matrix.append([0]*size)
        
        counter = 0 # tracks location of 1
        # iterate through incrementing 1 through each row 
        for i in range(len(id_matrix)):
            id_matrix[i][counter] = 1
            counter += 1
            
        return id_matrix
    
    @staticmethod
    def lu_decomposition(matrix):
        """
        Method -- lu_decomposition
        Given a square matrix, This method performs LU decomposition on it.
        LU decomposition is a factorization of a matrix as the product of a lower triangular matrix and an upper 
        triangular matrix. The lower triangular matrix L has the property that all the elements above the main 
        diagonal are zero, and the upper triangular matrix U has all the elements below the main diagonal equal to zero.
        The function returns a tuple of two matrices, L and U.

        Parameters:
        matrix (list of lists of float) -- The matrix to perform LU decomposition on.

        Returns:
        tuple -- A tuple of two matrices, L and U.
        """
        # number of rows
        m = len(matrix)
        # number of columns
        n = len(matrix[0])
        # Set identity matrix to L
        L = BasicLinAlg.identity(m)
        # Get U matrix
        U = [[0] * n for entries in range(m)]

        
        # loop through matrix rows
        for i in range(m):
            # loop through column entries in row
            for j in range(n):
                # perform row operations
                U[i][j] = matrix[i][j] - sum(L[i][k] * U[k][j] for k in range(i))
                
            # add 1's along pivot
            for j in range(i, n): 
                if i == j:
                    L[i][i] = 1
                else:
                    # input row operation values in L
                    L[j][i] = (matrix[j][i] - sum(L[j][k] * U[k][i] for k in range(i))) / U[i][i]
                    
        return L, U
    
    @staticmethod
    def qr_decomposition(matrix):
        """
        Method -- qr_decomposition
        This method computes the QR decomposition of a given matrix using the Gram-Schmidt process.
        The QR decomposition factorizes the matrix into the product of a matrix Q with orthonormal columns
        and an upper triangular matrix R. If the input matrix is square and has linearly independent columns, 
        the matrix Q will be orthogonal. For rectangular matrices, Q will have orthonormal columns but will not 
        be square.

        Parameters:
        matrix (list of list) -- The input matrix to be factorized. The matrix must have more rows 
                                        than columns (or be square) and have linearly independent columns.

        Returns:
        Q (list of list) -- The matrix Q from the QR decomposition. Q has orthonormal columns, and it 
                                    will be a square orthogonal matrix if the input matrix is square and has 
                                    linearly independent columns.
        R (list of list) -- The upper triangular matrix R from the QR decomposition. Its dimensions 
                                    will match the dimensions of the input matrix's columns.
        """

        rows, cols = BasicLinAlg.shape(matrix)

        if cols > rows:
            raise ValueError("The input matrix must have more rows than columns or be square.")

        Q = [[0 for _ in range(rows)] for _ in range(rows)]
        R = [[0 for _ in range(cols)] for _ in range(min(rows,cols))]

        for i in range(min(rows, cols)):
            v = [[matrix[k][i]] for k in range(rows)]

            for j in range(i):
                R[j][i] = BasicLinAlg.dot_product([Q[j]], v)
                v = [[v[k][0] - R[j][i] * Q[j][k]] for k in range(rows)]

            R[i][i] = BasicLinAlg.vector_norm(v)
            Q[i] = [v[k][0] / R[i][i] for k in range(rows)]

        Q = BasicLinAlg.transpose(Q)
        return Q, R

    @staticmethod
    def is_symmetric(matrix):
        """
        Method -- is_symmetric
        This method checks whether a given matrix is symmetric or not.
        A matrix is symmetric if it is equal to its transpose.

        Parameters:
        matrix (list of list) -- The input matrix to be checked.

        Returns:
        bool -- True if the matrix is symmetric, False otherwise.
        """
        
        #if matrix = its transpose: symmetric
        if matrix == BasicLinAlg.transpose(matrix):
            return True
        else:
            return False
    
    @staticmethod
    def eigenvalues(matrix):
        """
        Method -- eigenvalues
        This method calculates the eigenvalues of a given square matrix using the characteristic polynomial method.

        Parameters:
        matrix (list of lists of float or int): A square matrix for which the eigenvalues need to be calculated.

        Returns:
        list of float or int: A list containing the eigenvalues of the input matrix.
        Note: This implementation relies on NumPy for calculating the characteristic polynomial and its roots.
        """
        # Check for empty matrix
        if not matrix: 
            raise ValueError("Matrix must be non-empty")

        if len(matrix) != len(matrix[0]):
            raise ValueError("Matrix must be square.")

        # Calculate the characteristic polynomial of the matrix.
        poly = np.poly(matrix)

        # Find the roots of the characteristic polynomial to get the eigenvalues.
        eigenvalues = np.roots(poly)

        # formatted_eigenvalues = [float(format(eigenvalue, ".14f")) for eigenvalue in eigenvalues]
        return eigenvalues

    @staticmethod
    def null_space(matrix):
        """
        Method -- null_space
        This method returns the null space (also known as the kernel) of a given matrix as a list of column vectors
        that form a basis for the null space. The null space is the set of all vectors that, when multiplied by the matrix,
        result in the zero vector.

        The method calculates the null space by finding the reduced row echelon form (RREF) of the matrix and then
        solving the corresponding homogeneous system of linear equations.

        Note: In the case where the null space only contains the zero vector (i.e., the matrix has full column rank),
        this method returns an empty list. This is a valid representation of the trivial null space, and any operations
        performed on the null space should account for the possibility of an empty list.

        Parameters:
        matrix (list of lists of float or int): A 2D list representing the input matrix.

        Returns:
        list of lists of float or int: A list of column vectors forming a basis for the null space of the input matrix.
        """

        if not matrix or not matrix[0]:
            raise ValueError("Cannot compute the null space of an empty matrix")

        rref_matrix = BasicLinAlg.rref(matrix)

        pivot_cols = []
        free_vars = set(range(len(matrix[0])))

        for r, row in enumerate(rref_matrix):
            pivot = None
            for c, elem in enumerate(row):
                if abs(elem) > BasicLinAlg.eps:
                    pivot = c
                    pivot_cols.append(c)
                    free_vars.remove(c)
                    break
            if pivot is None:
                break

        basis_vectors = []
        for free_var in free_vars:
            vector = [0] * len(matrix[0])
            vector[free_var] = 1
            for i, row in enumerate(rref_matrix):
                if i < len(pivot_cols) and row[free_var] != 0:
                    vector[pivot_cols[i]] = -row[free_var]
            basis_vectors.append(vector)

        result =  basis_vectors
        return result

    @staticmethod
    def eigenvectors(matrix):
        """
        Method -- eigenvectors
        This method calculates the eigenvectors of a given square matrix. Each eigenvector in the output list corresponds
        to the eigenvalue in the same position in the list of eigenvalues returned by the eigenvalues method.

        The method calculates the eigenvectors by first finding the eigenvalues of the matrix. Then, for each eigenvalue,
        it constructs a matrix with the eigenvalue subtracted from the main diagonal elements and calculates the null space
        of this matrix, which gives the eigenvectors corresponding to the eigenvalue. The eigenvectors are then normalized
        to have a magnitude of 1.

        Parameters:
        matrix (list of lists of float or int): A square matrix.

        Returns:
        list of lists of float or int: A list of lists representing the eigenvectors of the matrix. Each eigenvector in the
        list corresponds to the eigenvalue in the same position in the list of eigenvalues returned by the eigenvalues method.
        """

        if not matrix:
            raise ValueError("Matrix must be non-empty")

        if not all(len(row) == len(matrix) for row in matrix):
            raise ValueError("Matrix must be square")
        
        if not all(isinstance(element, (float, int)) for row in matrix for element in row):
            raise ValueError("Matrix elements must be numbers")

        eigenvalues = BasicLinAlg.eigenvalues(matrix)
        eigenvectors = []

        for eigenvalue in eigenvalues:
            eigen_matrix = [[matrix[i][j] - (eigenvalue if i == j else 0) for j in range(len(matrix))] for i in range(len(matrix))]
            eigenvectors_for_eigenvalue = BasicLinAlg.null_space(eigen_matrix)

            for eigenvector in eigenvectors_for_eigenvalue:
                magnitude = sum([abs(elem)**2 for elem in eigenvector])**0.5
                eigenvector = [elem / magnitude for elem in eigenvector]
                eigenvectors.append(eigenvector)

        return eigenvectors
