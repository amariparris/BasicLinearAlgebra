"""
File: test.py

This script demonstrates the usage of the BasicLinearAlgebraPackage by importing the BasicLinAlg
class and performing a sample operation (eigenvector computation) on a test matrix.

The purpose of this script is to show how users can import and use the BasicLinAlg class in their
own projects. Replace the sample operation with any other desired operation available in the
BasicLinAlg class to test different functionalities.
"""

import BasicLinearAlgebraPackage as blap

lin_alg = blap.BasicLinAlg()

# Test any function from the BasicLinAlg class
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
eigenvalues = lin_alg.eigenvectors(matrix)
print(eigenvalues)