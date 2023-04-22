# BasicLinearAlgebra
This repository contains a Python implementation of a basic linear algebra package, with a collection of static methods for performing various operations on arrays and matrices. 
The implementation is designed to be easy to use and understand while providing a wide range of linear algebra operations.

## Features
The BasicLinAlg class in BasicLinearAlgebraPackage.py provides the following static methods:

Basic arithmetic operations on vectors: addition, subtraction, scalar multiplication, dot product, and vector norm.
Matrix operations: matrix multiplication, transposition, determinant, inverse, matrix addition and subtraction, matrix power, rank, shape, reduced row echelon form, LU decomposition, QR decomposition, eigenvalues, and eigenvectors.
The input arrays and matrices are expected to be represented as Python lists, with each element of the list representing a row or column of the array/matrix.

The implementation uses the numpy library only for calculating the characteristic polynomial and its roots when computing eigenvalues.

## Usage
To use the BasicLinAlg class, simply import it into your Python script:
```
from BasicLinearAlgebraPackage import BasicLinAlg
```
You can then call the static methods of the class directly, for example:
```
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
eigenvalues = BasicLinAlg.eigenvalues(matrix)
```

## Testing
A comprehensive test suite is provided in the BasicLinearAlgebraPackage_testing.py file. 
The test suite covers a wide range of test cases, including edge cases, to ensure the correctness and stability of the implemented methods. 
To run the tests, simply execute the BasicLinearAlgebraPackage_testing.py script:
```
python BasicLinearAlgebraPackage_testing.py
```

## Dependencies
The package has the following dependency:
* numpy

To install the dependency, use the following command:
```
pip install numpy
```

## Contributing
Contributions to the repository are welcome. If you find a bug or have a suggestion for an improvement, please open an issue or submit a pull request.
