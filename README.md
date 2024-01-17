# DCON
#### A Globally Convergent Algorithm for Neural Network Parameter Optimization Based on Difference-of-Convex Functions

This package contains an implemantation of the DCON algorithm introduced in
> D. Tschernutter, M. Kraus, S. Feuerriegel (2024). A Globally Convergent Algorithm for Neural Network Parameter Optimization Based on Difference-of-Convex Functions. Transactions on Machine Learning Research (TMLR)

Within the package we use the OSQP solver for our subproblems introduced in
> B. Stellato, G. Banjac, P. Goulart, A. Bemporad, S. Boyd (2020). OSQP: An operator splitting solver for quadratic programs. Mathematical Programming Computation

All details about this solver including licensing information can be found on https://osqp.org/

#### Instructions

The package can be installed via
```
pip install .
```
and used as follows. Note that we provide a method .get_keras() that allows to extract a Keras model using the optimized neural network weights after training.

```python
import numpy as np
from DCON import DCON

X_Train = np.array(((0,0),(1,1),(2,2)), dtype='d')
Y_Train = np.array((0,1,2), dtype='d')

X_Test = np.array(((0.5,0.5),(1.5,1.5)), dtype='d')
Y_Test = np.array((0.5,1.5), dtype='d')

model = DCON(n_hidden=10, n_inputs=X_Train.shape[1])
model.fit(X_Train, Y_Train, n_epochs='auto')

keras_model = model.get_keras()
test_loss = keras_model.evaluate(X_Test, Y_Test, verbose=0)
print("Loss: {}".format(test_loss))
```
