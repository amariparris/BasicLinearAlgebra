import unittest
import numpy as np
from BasicLinearAlgebraPackage import BasicLinAlg

class TestBasicLinAlg(unittest.TestCase):
    """
    This class is a collection of test cases for the BasicLinAlg class.

    The purpose of this class is to test the functionality of the BasicLinAlg class, ensuring
    that its methods perform correctly under various circumstances. The test cases cover normal cases,
    edge cases, and error handling for all the methods in the BasicLinAlg class.

    Each test method in this class corresponds to one method in the BasicLinAlg class, and
    multiple test cases are provided within each test method. The test cases are designed to cover a
    range of inputs, including valid and invalid inputs, as well as different types of input data.
    """
    
    def test_vector_norm(self):
        """
        Method -- test_vector_norm
        This method tests the vector_norm method in the BasicLinAlg class.

        It tests the following cases:
        1. Normal case: Compute the Euclidean norm of a non-empty row vector.
        2. Normal case: Compute the Euclidean norm of a non-empty column vector.
        3. Edge case: Compute the Euclidean norm of an empty vector.
        4. Edge case: Compute the Euclidean norm of a row vector with negative elements.
        5. Edge case: Compute the Euclidean norm of a column vector with negative elements.
        6. Edge case: Compute the Euclidean norm of a row vector with floating-point elements.
        7. Edge case: Compute the Euclidean norm of a column vector with floating-point elements.
        """
        # Test case 1
        vector = [[1, 2, 3]]
        expected_result = 3.7416573867739413
        self.assertAlmostEqual(BasicLinAlg.vector_norm(vector), expected_result)

        # Test case 2
        vector = [[1], [2], [3]]
        expected_result = 3.7416573867739413
        self.assertAlmostEqual(BasicLinAlg.vector_norm(vector), expected_result)

        # Test case 3
        vector = []
        expected_result = 0
        self.assertEqual(BasicLinAlg.vector_norm(vector), expected_result)

        # Test case 4
        vector = [[-1, -2, -3]]
        expected_result = 3.7416573867739413
        self.assertAlmostEqual(BasicLinAlg.vector_norm(vector), expected_result)

        # Test case 5
        vector = [[-1], [-2], [-3]]
        expected_result = 3.7416573867739413
        self.assertAlmostEqual(BasicLinAlg.vector_norm(vector), expected_result)

        # Test case 6
        vector = [[1.5, 2.5, 3.5]]
        expected_result = 4.55521678957215
        self.assertAlmostEqual(BasicLinAlg.vector_norm(vector), expected_result)

        # Test case 7
        vector = [[1.5], [2.5], [3.5]]
        expected_result = 4.55521678957215
        self.assertAlmostEqual(BasicLinAlg.vector_norm(vector), expected_result)


    def test_dot_product(self):
        """
        Method -- test_dot_product
        This method tests the dot_product method in the BasicLinAlg class.

        It tests the following cases:
        1. Normal case: Dot product of two vectors of the same length.
        2. Edge case: Dot product of two vectors with different lengths, raising ValueError.
        3. Edge case: Dot Product of two empty vectors.
        4. Edge case: Dot Product of two vectors with negative elements.
        5. Edge case: Dot Product of column and row vectors
        5. Edge case: Dot Product of a vector with floating-point elements.
        """
        # Test case 1
        v1 = [[1, 2, 3]]
        v2 = [[4, 5, 6]]
        expected_result = 32
        self.assertEqual(BasicLinAlg.dot_product(v1, v2), expected_result)

        # Test case 2
        v1 = [[1, 2, 3]]
        v2 = [[4, 5]]
        with self.assertRaises(ValueError):
            BasicLinAlg.dot_product(v1, v2)

        # Test case 3
        v1 = []
        v2 = []
        expected_result = 0
        self.assertEqual(BasicLinAlg.dot_product(v1, v2), expected_result)

        # Test case 4
        v1 = [[-1], [-2], [-3]]
        v2 = [[-4], [-5], [-6]]
        expected_result = 32
        self.assertEqual(BasicLinAlg.dot_product(v1, v2), expected_result)

        # Test case 5
        v1 = [[-1, -2, -3]]
        v2 = [[4], [5], [6]]
        expected_result = -32
        self.assertEqual(BasicLinAlg.dot_product(v1, v2), expected_result)

        # Test case 6
        v1 = [[1.5, 2.5, 3.5]]
        v2 = [[4.5, 5.5, 6.5]]
        expected_result = 43.25
        self.assertAlmostEqual(BasicLinAlg.dot_product(v1,v2), expected_result)

    def test_transpose(self):
        """
        Method -- test_transpose
        This method tests the transpose method in the BasicLinAlg class.

        It tests the following cases:
        1. Normal case: transpose of 2x2 matrix.
        2. Edge case: transpose of 2x4 matrix. 
        3. Edge case: transpose of 4x2 matrix
        """
        
        # Test 1
        m1 = [[1,2],[3,4]]
        expected_result = [[1,3],[2,4]]
        self.assertEqual(BasicLinAlg.transpose(m1), expected_result)    

        # Test 2
        m2 = [[1,2,3,4],[5,6,7,8]]
        expected_result = [[1,5],[2,6],[3,7],[4,8]]
        self.assertEqual(BasicLinAlg.transpose(m2), expected_result)

        # Test 3
        m3 = [[1,2],[3,4],[5,6],[7,8]]
        expected_result = [[1,3,5,7], [2,4,6,8]]
        self.assertEqual(BasicLinAlg.transpose(m3), expected_result)
    
    def test_get_submatrix(self):
        """
        Method -- test_get_submatrix
        This method tests the get_submatrix method in the BasicLinAlg class.

        It tests the following cases:
        1. Normal case: Get a submatrix from a 3x3 matrix.
        2. Normal case: Get a submatrix from a 2x2 matrix.
        3. Edge case: Get a submatrix from a 1x1 matrix.
        """
        # Test case 1
        matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        expected_result = [[5, 6], [8, 9]]
        self.assertEqual(BasicLinAlg.get_submatrix(matrix, row=1, col=1), expected_result)

        # Test case 2
        matrix = [[1, 2], [3, 4]]
        expected_result = [[4]]
        self.assertEqual(BasicLinAlg.get_submatrix(matrix, row=1, col=1), expected_result)

        # Test case 3
        matrix = [[1]]
        expected_result = []
        self.assertEqual(BasicLinAlg.get_submatrix(matrix, row=1, col=1), expected_result)

    def test_inverse(self):
        """
        Method -- test_inverse
        This method tests the inverse method in the BasicLinAlg class.

        It tests the following cases:
        1. Normal case: Inverse of a 2x2 square matrix.
        2. Normal case: Inverse of a 3x3 square matrix.
        3. Edge case: Inverse of a singular matrix, raising ValueError.
        4. Edge case: Inverse of a non-square matrix, raising ValueError.
        5. Edge case: Inverse of an empty matrix, raising ValueError.
        """
        # Test Case 1
        matrix = [[1, 2], [3, 4]]
        expected_result = [[-2, 1], [1.5, -0.5]]
        self.assertEqual(BasicLinAlg.inverse(matrix), expected_result)

        # Test Case 2
        matrix = [[1, 0, 1], [0, 1, 2], [1, 2, 3]]
        expected_result = [[0.5, -1.0, 0.5], [-1.0, -1.0, 1.0], [0.5, 1.0, -0.5]]
        self.assertEqual(BasicLinAlg.inverse(matrix), expected_result)

        # Test Case 3
        matrix = [[1, 2], [2, 4]]
        with self.assertRaises(ValueError):
            BasicLinAlg.inverse(matrix)

        # Test Case 4
        matrix = [[1, 2, 3], [4, 5, 6]]
        with self.assertRaises(ValueError):
            BasicLinAlg.inverse(matrix)

        # Test Case 5
        matrix = []
        with self.assertRaises(ValueError):
            BasicLinAlg.inverse(matrix)

    
    def test_determinant(self):
        """
        Method -- test_determinant
        This method tests the determinant method in the BasicLinAlg class.

        It tests the following cases:
        1. Normal case: Compute the determinant of a 2x2 square matrix.
        2. Normal case: Compute the determinant of a 3x3 square matrix.
        3. Edge case: Compute the determinant of a 1x1 square matrix.
        4. Error case: Attempt to compute the determinant of a non-square matrix.
        """
        # Test case 1
        matrix = [[1, 2], [3, 4]]
        expected_result = -2
        self.assertEqual(BasicLinAlg.determinant(matrix), expected_result)

        # Test case 2
        matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        expected_result = 0
        self.assertEqual(BasicLinAlg.determinant(matrix), expected_result)

        # Test case 3
        matrix = [[5]]
        expected_result = 5
        self.assertEqual(BasicLinAlg.determinant(matrix), expected_result)

        # Test case 4
        matrix = [[2, 3], [5, 6], [7, 8]]
        with self.assertRaises(ValueError):
            BasicLinAlg.determinant(matrix)

    def test_adjoint(self):
        """
        Method -- test_adjoint
        This method tests the adjoint method in the BasicLinAlg class.

        It tests the following cases:
        1. Normal case: Compute the adjoint of a 2x2 square matrix.
        2. Normal case: Compute the adjoint of a 3x3 square matrix.
        3. Edge case: Compute the adjoint of a 1x1 square matrix.
        4. Error case: Attempt to compute the adjoint of a non-square matrix.
        """
        # Test case 1
        matrix = [[1, 2], [3, 4]]
        expected_result = [[4, -2], [-3, 1]]
        self.assertEqual(BasicLinAlg.adjoint(matrix), expected_result)

        # Test case 2
        matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        expected_result = [[-3, 6, -3], [6, -12, 6], [-3, 6, -3]]
        self.assertEqual(BasicLinAlg.adjoint(matrix), expected_result)

        # Test case 3
        matrix = [[5]]
        expected_result = [[1]]
        self.assertEqual(BasicLinAlg.adjoint(matrix), expected_result)

        # Test case 4
        matrix = [[2, 3, 7], [5, 6, 8]]
        with self.assertRaises(ValueError):
            BasicLinAlg.adjoint(matrix)

    
    def test_scalar_multiply(self):
        """
        Method -- test_scalar_multiply
        This method tests the scalar_multiply method in the BasicLinAlg class.

        It tests the following cases:
        1. Normal case: Product of positive scalar and non-zero matrix
        2. Edge case: Product of negative scalar and non-zero matrix
        3. Edge case: Product of floating point scalar and non-zero matrix
        4. Edge case: Product of scalar and zero matrix
        5. Edge case: Product of scalar and empty matrix
        """
        # Test Case 1
        matrix = [[1,2],[3,4], [5,6]]
        scalar = 2
        expected_result = [[2,4],[6,8], [10, 12]]
        self.assertEqual(BasicLinAlg.scalar_multiply(matrix, scalar), expected_result)   

        # Test Case 2
        matrix = [[1,2],[3,4], [5,6]]
        scalar = -2
        expected_result = [[-2,-4],[-6,-8], [-10, -12]]
        self.assertEqual(BasicLinAlg.scalar_multiply(matrix, scalar), expected_result)

        # Test Case 3
        matrix = [[1,2],[3,4], [5,6]]
        scalar = .5
        expected_result = [[.5,1],[1.5,2], [2.5, 3]]
        self.assertEqual(BasicLinAlg.scalar_multiply(matrix, scalar), expected_result)

       # Test Case 4
        matrix = [[0,0],[0,0], [0,0]]
        scalar = 3
        expected_result = [[0,0],[0,0], [0,0]]
        self.assertEqual(BasicLinAlg.scalar_multiply(matrix, scalar), expected_result)

        # Test Case 5
        matrix = []
        scalar = 2
        expected_result = []
        self.assertEqual(BasicLinAlg.scalar_multiply(matrix, scalar), expected_result)  
    
    def test_matrix_addition(self):
        """
        Method -- test_matrix_addition
        This method tests the matrix_addition method in the BasicLinAlg class.

        It tests the following cases:
        1. Normal case: Addition of two matrices of the same length.
        2. Edge case: Addition of two matrices with different lengths, raising ValueError.
        3. Edge case: Addition of two empty matrices.
        4. Edge case: Addition of two matrices with negative elements.
        """
        # Test case 1
        m1 = [[1, 2, 3],[1,2,3]]
        m2 = [[4, 5, 6],[1,1,1]]
        expected_result = [[5, 7, 9],[2,3,4]]
        self.assertEqual(BasicLinAlg.matrix_addition(m1, m2), expected_result)

        # Test case 2
        m1 = [[1, 2, 3]]
        m2 = [[4, 5, 6],[1,1,1]]
        with self.assertRaises(ValueError):
            BasicLinAlg.matrix_addition(m1, m2)

        # Test case 3
        m1 = [[],[]]
        m2 = [[],[]]
        expected_result = [[],[]]
        self.assertEqual(BasicLinAlg.matrix_addition(m1, m2), expected_result)

        # Test case 4
        m1 = [[-1, 2, -3],[0,-4,7]]
        m2 = [[4, -5, 6],[-2,4,-6]]
        expected_result = [[3, -3, 3],[-2,0,1]]
        self.assertEqual(BasicLinAlg.matrix_addition(m1, m2), expected_result)

    
    def test_matrix_subtraction(self):
        """
        Method -- test_matrix_subtraction
        This method tests the matrix_subtraction method in the BasicLinAlg class.

        It tests the following cases:
        1. Normal case: Subtraction of two matrices of the same length.
        2. Edge case: Subtraction of two matrices with different lengths, raising ValueError.
        3. Edge case: Subtraction of two empty matrices.
        4. Edge case: Subtraction of two matrices with negative elements.
        """
        # Test case 1
        m1 = [[1, 2, 3],[1,2,3]]
        m2 = [[4, 5, 6],[1,1,1]]
        expected_result = [[-3, -3, -3],[0,1,2]]
        self.assertEqual(BasicLinAlg.matrix_subtraction(m1, m2), expected_result)

        # Test case 2
        m1 = [[1, 2, 3]]
        m2 = [[4, 5, 6],[1,1,1]]
        with self.assertRaises(ValueError):
            BasicLinAlg.matrix_subtraction(m1, m2)

        # Test case 3
        m1 = [[],[]]
        m2 = [[],[]]
        expected_result = [[],[]]
        self.assertEqual(BasicLinAlg.matrix_subtraction(m1, m2), expected_result)

        # Test case 4
        m1 = [[-1, 2, -3],[0,-4,7]]
        m2 = [[4, -5, 6],[-2,4,-6]]
        expected_result = [[-5, 7, -9],[2,-8,13]]
        self.assertEqual(BasicLinAlg.matrix_subtraction(m1, m2), expected_result)


    def test_matrix_multiply(self):
        """
        Method -- test_matrix_multiply
        This method tests the matrix_multiply method in the BasicLinAlg class.

        It tests the following cases:
        1. Normal case: Dot product of two matrices of the appropriate size.
        2. Edge case: Product of two matrices with incorrect sizes, raising ValueError.
        3. Edge case: Product of two matrices with negative elements.
        4. Edge case: Product of two matrices with all zeros.
        5. Edge case: Product of two 1x1 matrices.
        6. Edge case: Product of two matrices with floating-point elements.
        """
        # Test case 1 
        m1 = [[1, 2, 3], [4, 5, 6]]
        m2 = [[2, 3], [1, 7], [3, 8]]
        expected = [[13, 41], [31, 95]] 
        self.assertEqual(BasicLinAlg.matrix_multiply(m1, m2), expected)

        # Test case 2
        m1 = [[1, 2, 3],[4,4,5]] 
        m2 = [[2,3,1],[4,4,5]]
        with self.assertRaises(ValueError):
            BasicLinAlg.matrix_multiply(m1, m2)


        # Test case 3
        m1 = [[-1, -2],[5, -7]]
        m2 = [[-4, -5],[-6, 0]]
        expected = [[16, 5], [22, -25]]
        self.assertEqual(BasicLinAlg.matrix_multiply(m1, m2), expected)

        # test case 4
        m1 = [[0],[0]]
        m2 = [[0,0,0]]
        expected = [[0, 0, 0], [0, 0, 0]]
        self.assertEqual(BasicLinAlg.matrix_multiply(m1, m2), expected)
        

        # test case 5
        m1 = [[1]]
        m2 = [[2]]
        expected = [[2]]
        self.assertEqual(BasicLinAlg.matrix_multiply(m1, m2), expected)
        
        # Test case 6
        m1 = [[1.5, 2.5],[3.2, 1.1]]
        m2 = [[4.5, 5.5],[7.7, 3.2]]
        expected = [[26.0, 16.25], [22.87, 21.12]]
        self.assertAlmostEqual(BasicLinAlg.matrix_multiply(m1,m2), expected)

    
    def test_matrix_power(self):
        """
        Method -- test_matrix_power
        This method tests the matrix_power method in the BasicLinAlg class.

        It tests the following cases:
        1. Normal case: Square Matrix raised to positive power.
        2. Edge case: Matrix raised to zero. 
        3. Edge case: Matrix raised to negative power.
        4. Edge case: Matrix raised to a floating point power.
        5. Edge case: Non-Square Matrix raised to positive power.
        """
        # Test case 1 
        matrix = [[1,2],[3,4]]
        power = 3
        expected = [[37, 54], [81, 118]] 
        self.assertEqual(BasicLinAlg.matrix_power(matrix, power), expected)

        # Test case 2
        matrix = [[1,2],[3,4]]
        power = 0
        with self.assertRaises(ValueError):
            BasicLinAlg.matrix_power(matrix, power)

        # Test case 3
        matrix = [[1,2],[3,4]]
        power = -4
        with self.assertRaises(ValueError):
            BasicLinAlg.matrix_power(matrix, power)

        # test case 4
        matrix = [[1,2],[3,4]] 
        power = -.5
        with self.assertRaises(ValueError):
            BasicLinAlg.matrix_power(matrix, power)

        # test case 5
        matrix = [[1, 2, 3],[4,4,5]] 
        power = 3
        with self.assertRaises(ValueError):
            BasicLinAlg.matrix_power(matrix, power)
    
    def test_identity(self):
        """
        Method -- test_identity
        This method tests the test_identity method in the BasicLinAlg class.

        It tests the following cases:
        1. Normal case: Identity matrix with size 2.
        2. Normal case: Identity matrix with size 3.
        3. Normal case: Identity matrix with size 4.
        4. Edge case: Size given as float raises TypeError. 
        """

        # test 1 size 2
        size_2 = [[1,0],[0,1]]
        self.assertEqual(BasicLinAlg.identity(2), size_2)

        # test 2 size 0 
        size_0 = [[]]
        self.assertEqual(BasicLinAlg.identity(0), size_0)

        # test 3 size 4
        size_4 = [[1,0,0,0], [0,1,0,0],[0,0,1,0],[0,0,0,1]]
        self.assertEqual(BasicLinAlg.identity(4), size_4)

        # test 4 type error
        with self.assertRaises(TypeError):
            BasicLinAlg.identity(2.3)

    def test_rref(self):
        """
        Method -- test_rref
        This method tests the rref method in the BasicLinAlg class.

        It tests the following cases:
        1. Normal case: Reduced row echelon form of square matrix.
        2. Normal case: Reduced row echelon form of vertical matrix. 
        3. Normal case: Reduced row echelon form of horizontal matrix.
        4. Edge case: Reduced row echelon form of zero matrix.
        5. Edge case: Reduced row echelon form of matrix with floating-point entries.
        """
        # Test Case 1
        matrix = [[1,2],[3,4]]
        expected_result = [[1,0],[0,1]]
        self.assertAlmostEqual(BasicLinAlg.rref(matrix), expected_result)   

        # Test Case 2
        matrix = [[1,2],[3,4], [5,6]]
        expected_result = [[1,0],[0,1], [0, 0]]
        self.assertAlmostEqual(BasicLinAlg.rref(matrix), expected_result)

        # Test Case 3
        matrix = [[1, 2, 3], [4, 5, 6]]
        expected_result = [[1, 0, -1], [0, 1, 2]]
        self.assertAlmostEqual(BasicLinAlg.rref(matrix), expected_result)

        # Test Case 4
        matrix = [[0,0],[0,0], [0,0]]
        expected_result = [[0,0],[0,0], [0,0]]
        self.assertAlmostEqual(BasicLinAlg.rref(matrix), expected_result)

        # Test Case 5
        matrix = [[.5,1],[1.5,2], [2.5, 3]]
        expected_result = [[1,0],[0,1], [0, 0]]
        self.assertAlmostEqual(BasicLinAlg.rref(matrix), expected_result)

    def test_rank(self):
        """
        Method -- test_rank
        This method tests the rank method in the BasicLinAlg class.

        It tests the following cases:
        1. Normal case: Rank of square matrix.
        2. Normal case: Rank of vertical matrix.
        3. Normal case: Rank of horizontal matrix.
        4. Edge case: Rank of zero matrix.
        5. Edge case: Rank of matrix with floating-point entries.
        """
        # Test Case 1
        matrix = [[1, 2], [3, 4]]
        expected_result = 2
        self.assertEqual(BasicLinAlg.rank(matrix), expected_result)

        # Test Case 2
        matrix = [[1, 2], [3, 4], [5, 6]]
        expected_result = 2
        self.assertEqual(BasicLinAlg.rank(matrix), expected_result)

        # Test Case 3
        matrix = [[1, -2, 3], [4, -5, 6]]
        expected_result = 2
        self.assertEqual(BasicLinAlg.rank(matrix), expected_result)

        # Test Case 4
        matrix = [[0, 0], [0, 0], [0, 0]]
        expected_result = 0
        self.assertEqual(BasicLinAlg.rank(matrix), expected_result)

        # Test Case 5
        matrix = [[0.5, 1], [1.5, 2], [2.5, 3]]
        expected_result = 2
        self.assertEqual(BasicLinAlg.rank(matrix), expected_result)

    def test_lu_decomposition(self):
        """
        Method -- test_lu_decomposition
        This method tests the lu_decomposition method in the BasicLinAlg class.

        It tests the following cases:
        1. Normal case: Non Square Matrix LU decomposition
        2. Normal case: Square Matrix LU decomposition
        3. Edge case:  Identity Matrix LU decomposition
        4. Edge case: Vector LU decomposition
        """
        # Test Case 1
        matrix = [[3,4],[-5,3], [5,4]]
        expected_result = ([[1, 0, 0], [-1.6666666666666667, 1, 0], [0, 0, 1]], [[3, 4], [0.0, 9.666666666666668], [5.0, 4.0]])
        self.assertAlmostEqual(BasicLinAlg.lu_decomposition(matrix), expected_result)   

        # Test Case 2
        matrix = [[1,2,3],[4,5,6],[7,8,9]]
        expected_result = ([[1, 0, 0], [4.0, 1, 0], [7.0, 2.0, 1]], [[1, 2, 3], [0.0, -3.0, -6.0], [0.0, 0.0, 0.0]])
        self.assertAlmostEqual(BasicLinAlg.lu_decomposition(matrix), expected_result) 

        # Test Case 3
        matrix = [[1,0,0],[0,1,0],[0,0,1]]
        expected_result = ([[1, 0, 0], [0.0, 1, 0], [0.0, 0.0, 1]], [[1, 0, 0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        self.assertAlmostEqual(BasicLinAlg.lu_decomposition(matrix), expected_result) 

        # Test Case 4
        matrix = [[1],[2],[3]]
        expected_result = ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1], [2], [3]])
        self.assertAlmostEqual(BasicLinAlg.lu_decomposition(matrix), expected_result) 

    def test_qr_decomposition(self):
        """
        Method -- test_qr_decomposition
        This method tests the qr_decomposition method in the BasicLinAlg class.

        It tests the following cases:
        1. Normal case: QR decomposition of a 3x2 matrix.
        2. Edge case: QR decomposition of a square matrix.
        3. Edge case: QR decomposition of a rectangular matrix with more columns than rows, raising ValueError.
        """

        # Test case 1
        m1 = [[1, 22], [4, 5], [7, 8]]
        expected_Q = [
            [0.12309149097933272, 0.9922345963916136, 0],
            [0.4923659639173309, -0.04543467132664696, 0],
            [0.8616404368553291, -0.1157851301550036, 0]
        ]
        expected_R = [
            [8.12403840463596, 12.062966115974607],
            [0, 20.675706722742238],
            [0, 0]
        ]

        Q, R = BasicLinAlg.qr_decomposition(m1)
        for i in range(len(Q)):
            for j in range(len(Q[0])):
                self.assertAlmostEqual(Q[i][j], expected_Q[i][j])
        for i in range(len(R)):
            for j in range(len(R[0])):
                self.assertAlmostEqual(R[i][j], expected_R[i][j])

        # Test case 2
        m2 = [[4, 7, -8], [.75, 1.54, -9], [7, -4.8, 24.8]]
        expected_Q = [
            [0.49400601687555074, 0.8446299641591886, 0.20629658003758006], 
            [0.09262612816416577, 0.18479477942765551, -0.9784024171462371], 
            [0.8645105295322139, -0.5024451344582136, -0.01305492961294868]
        ]
        expected_R = [
            [8.097067370350823, -0.548964186252956, 16.654177843917005],
            [0, 8.608730354832332, -20.880832062686107], 
            [0, 0, 6.831486859614385]
        ]

        Q, R = BasicLinAlg.qr_decomposition(m2)
        for i in range(len(Q)):
            for j in range(len(Q[0])):
                self.assertAlmostEqual(Q[i][j], expected_Q[i][j])
        
        for i in range(len(R)):
            for j in range(len(R[0])):
                self.assertAlmostEqual(R[i][j], expected_R[i][j])

        # Test case 3
        m3 = [[1, 2, 200,4], [3, 4, 55,4]]
        with self.assertRaises(ValueError):
            BasicLinAlg.qr_decomposition(m3)
        
    def test_scalar_multiply_vector(self):
        """
        Method -- test_scalar_multiply_vector
        This method tests the scalar_multiply_vector method in the BasicLinAlg class.

        It tests the following cases:
        1. Normal case: Vector passed in as column * 5
        2. Normal case: Vector passed in as row * 5
        3. Edge case: Entry is not a vector
        4. Edge case: Empty vector
        """
         # Test Case 1
        v1 = [[1],[2],[3]]
        scalar = 5
        expected_result = [[5],[10],[15]]
        self.assertEqual(BasicLinAlg.scalar_multiply_vector(v1, scalar), expected_result)

        # Test Case 2
        v2 = [[1,2,3]]
        scalar = 5
        expected_result = [[5,10,15]]
        self.assertEqual(BasicLinAlg.scalar_multiply_vector(v2, scalar), expected_result)

        # Test Case 3
        v3 = [[1,2,3], [4,5,6]]
        scalar = 5
        with self.assertRaises(ValueError):
            BasicLinAlg.scalar_multiply_vector(v3, scalar)

        # Test Case 4
        v4 = []
        scalar = 5
        with self.assertRaises(ValueError):
            BasicLinAlg.scalar_multiply_vector(v4, scalar)        
        
    def test_is_symmetric(self):
        """
        Method -- test_is_symmetric
        This method tests the is_symmetric method in the BasicLinAlg class.

        It tests the following cases:
        1. Normal case: Symmetric square matrix.
        2. Normal case: Non-symmetric square matrix.
        3. Edge case: Empty matrix.
        4. Edge case: Non-square matrix.
        """
        # Test Case 1
        matrix = [[1, 2, 3], [2, 4, 5], [3, 5, 6]]
        self.assertTrue(BasicLinAlg.is_symmetric(matrix))

        # Test Case 2
        matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        self.assertFalse(BasicLinAlg.is_symmetric(matrix))

        # Test Case 3
        matrix = []
        self.assertTrue(BasicLinAlg.is_symmetric(matrix))

        # Test Case 4
        matrix = [[1, 2, 3], [4, 5, 6]]
        self.assertFalse(BasicLinAlg.is_symmetric(matrix))
    
    def test_shape(self):
        """
        Method -- test_shape
        This method tests the shape method in the BasicLinAlg class.

        It tests the following cases:
        1. Normal case: square matrix
        2. Normal case: Horizontal matrix
        3. Normal case: Vertical matrix
        4. Normal case: Vector
        5. Edge case: Empty Matrix
        """
        # Test Case 1
        matrix = [[1,2],[3,4]]
        expected_result = (2,2)
        self.assertEqual(BasicLinAlg.shape(matrix), expected_result) 

        # Test Case 2
        matrix = [[1,2],[3,4], [5,6]]
        expected_result = (3,2)
        self.assertEqual(BasicLinAlg.shape(matrix), expected_result)  

        # Test Case 3
        matrix = [[1,2,3],[4,5,6]]
        expected_result = (2,3)
        self.assertEqual(BasicLinAlg.shape(matrix), expected_result) 

        # Test Case 4
        matrix = [[1],[3], [5]]
        expected_result = (3,1)
        self.assertEqual(BasicLinAlg.shape(matrix), expected_result)

        # Test Case 5
        matrix = []
        expected_result = (0,0)
        self.assertEqual(BasicLinAlg.shape(matrix), expected_result)  

    def test_eigenvalues(self):
        """
        Method -- test_eigenvalues
        This method tests the eigenvalues method in the BasicLinAlg class.

        It tests the following cases:
        1. Normal case: Diagonal matrix
        2. Normal case: Symmetric matrix
        3. Normal case: General square matrix
        4. Edge case: Empty matrix
        5. Edge case: Non-square matrix
        """

        def check_eigenvalues(matrix, eigenvalues, tol=1e-6):
            """
            Helper function -- check_eigenvalues
            This helper function compares the eigenvalues of a given matrix calculated using the BasicLinAlg class method
            'eigenvalues' with the eigenvalues calculated using the NumPy library. It checks if the eigenvalues are equal
            within a specified tolerance level.

            The function first sorts both the eigenvalues calculated by the BasicLinAlg class method and the NumPy library
            to ensure that they are compared in the same order.

            Parameters:
            matrix (list of lists of float or int): A square matrix for which the eigenvalues are calculated.
            eigenvalues (list of float or int): A list of eigenvalues calculated using the BasicLinAlg class method.
            tol (float, optional): The tolerance level within which the eigenvalues are considered equal. Default is 1e-6.

            Returns:
            bool: True if all the eigenvalues match within the specified tolerance level, False otherwise.
            """
            numpy_eigenvalues = np.linalg.eigvals(matrix)
            numpy_eigenvalues.sort()  
            eigenvalues.sort()  

            return all(np.isclose(eigenvalue, numpy_eigenvalue, rtol=0, atol=tol)
                    for eigenvalue, numpy_eigenvalue in zip(eigenvalues, numpy_eigenvalues))

        # Test Case 1
        matrix = [[2, 0], [0, 3]]
        eigenvalues = BasicLinAlg.eigenvalues(matrix)
        self.assertTrue(check_eigenvalues(matrix, eigenvalues))

        # Test Case 2
        matrix = [[4, 2], [2, 3]]
        eigenvalues = BasicLinAlg.eigenvalues(matrix)
        self.assertTrue(check_eigenvalues(matrix, eigenvalues))

        # Test Case 3
        matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        eigenvalues = BasicLinAlg.eigenvalues(matrix)
        self.assertTrue(check_eigenvalues(matrix, eigenvalues))

        # Test Case 4
        with self.assertRaises(ValueError):
            BasicLinAlg.eigenvalues([])

        # Test Case 5
        with self.assertRaises(ValueError):
            BasicLinAlg.eigenvalues([[1, 2], [3, 4], [5, 6]])

    def test_eigenvectors(self):
        """
        Method -- test_eigenvectors
        This method tests the eigenvectors method in the BasicLinAlg class.

        It tests the following cases:
        1. Normal case: 1x1 matrix
        2. Normal case: 2x2 matrix
        3. Normal case: 3x3 matrix
        4. Edge case: Empty matrix
        """

        def check_eigenvectors(matrix, eigenvectors, eigenvalues):
            """
            Helper function -- check_eigenvectors
            This function checks if the eigenvectors satisfy the matrix equation Av = λv
            for a given matrix A, its eigenvectors, and eigenvalues.

            Parameters:
            matrix (list of lists of float or int): The input matrix A.
            eigenvectors (list of lists of float or int): The eigenvectors of matrix A.
            eigenvalues (list of float or int): The eigenvalues of matrix A.

            Returns:
            bool: True if the eigenvectors satisfy the matrix equation Av = λv for all eigenvectors and eigenvalues,
                False otherwise.
            """
            A = np.array(matrix)
            tolerance = 1e-6

            for i, eigenvector in enumerate(eigenvectors):
                eigenvalue = eigenvalues[i]
                Av = np.matmul(A, eigenvector)
                λv = [eigenvalue * elem for elem in eigenvector]

                if not np.allclose(Av, λv, atol=tolerance):
                    return False

            return True

        # Test Case 1
        matrix = [[3]]
        eigenvalues = BasicLinAlg.eigenvalues(matrix)
        eigenvectors = BasicLinAlg.eigenvectors(matrix)
        self.assertTrue(check_eigenvectors(matrix, eigenvectors, eigenvalues))

        # Test Case 2
        matrix = [[1, 2], [3, 4]]
        eigenvalues = BasicLinAlg.eigenvalues(matrix)
        eigenvectors = BasicLinAlg.eigenvectors(matrix)
        self.assertTrue(check_eigenvectors(matrix, eigenvectors, eigenvalues))

        # Test Case 3
        matrix = [[1, 2.5, 3], [2.6, 4, 5.3], [3, 5, 6]]
        eigenvalues = BasicLinAlg.eigenvalues(matrix)
        eigenvectors = BasicLinAlg.eigenvectors(matrix)
        self.assertTrue(check_eigenvectors(matrix, eigenvectors, eigenvalues))

        # Test Case 3
        matrix = [[10, 2.5, -3], [2.6, 4, 5.3], [3, -5, 6]]
        eigenvalues = BasicLinAlg.eigenvalues(matrix)
        eigenvectors = BasicLinAlg.eigenvectors(matrix)
        self.assertTrue(check_eigenvectors(matrix, eigenvectors, eigenvalues))

        # Test Case 4
        with self.assertRaises(ValueError):
            BasicLinAlg.eigenvectors([])


    def test_null_space(self):
        """
        Method -- test_null_space
        This method tests the null_space method in the BasicLinAlg class.

        It tests the following cases:
        1. Normal case: Matrix with floating-point values
        2. Edge case: Horizontal matrix
        3. Edge case: Empty matrix
        4. Edge case: Matrix with linearly dependent rows
        5. Edge case: Matrix with linearly dependent columns
        6. Edge case: Vertical matrix which has full column rank
        """
        # Test Case 1
        matrix = [[1.5, 0.5], [3.0, 1.0]]
        expected_result = [[-0.3333333333333333, 1]]
        result = BasicLinAlg.null_space(matrix)
        for i in range(len(expected_result)):
            for j in range(len(expected_result[0])):
                self.assertAlmostEqual(result[i][j], expected_result[i][j])

        # Test Case 2
        matrix = [[1, 2, 3], [4, 5, 6]]
        expected_result = [[1.0, -2.0, 1]]
        result = BasicLinAlg.null_space(matrix)
        for i in range(len(expected_result)):
            for j in range(len(expected_result[i])):
                self.assertAlmostEqual(result[i][j], expected_result[i][j])
        
        # Test Case 3
        with self.assertRaises(ValueError):
            BasicLinAlg.null_space([])

        # Test Case 4
        matrix = [[1, 2, 3], [2, 4, 6], [3, 6, 9]]
        expected_result = [[-2.0, 1, 0], [-3.0, 0, 1]]
        result = BasicLinAlg.null_space(matrix)
        for i in range(len(expected_result)):
            for j in range(len(expected_result[0])):
                self.assertAlmostEqual(result[i][j], expected_result[i][j])
        
        # Test Case 5
        matrix = [[1, 2, 1], [2, 4, 2], [3, 6, 3]]
        expected_result = [[-2.0, 1, 0], [-1.0, 0, 1]]
        result = BasicLinAlg.null_space(matrix)
        for i in range(len(expected_result)):
            for j in range(len(expected_result[0])):
                self.assertAlmostEqual(result[i][j], expected_result[i][j])
        
        # Test Case 6
        matrix = [[1, 4], [3, 5], [7, 6]] 
        expected_result = [] 
        result = BasicLinAlg.null_space(matrix)
        for i in range(len(expected_result)):
            for j in range(len(expected_result[i])):
                self.assertAlmostEqual(result[i][j], expected_result[i][j])

if __name__ == '__main__':
    unittest.main()
