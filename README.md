This is the code of EPFL master course Deep learning(mini project2), which is a basic framework for a neural network library.

In the code, the basic linear layer, the activation layer and the loss layer was implemented by pure pytorch matrix operator.

It is a great material if you want to learn the backpropagation.

Since the pytorch matrix operator supports GPU computation natively, this code can also perform GPU training.

There are two branches, the master branch is the fast version which is implemented by batch computation and a without_batch branch which is done by samples by samples. The batch compute version is way faster than the without_batch version, but the without batch version has a more intuitive math process of backpropagation. It is more easy to understand.

The pdf file is the report of the project, which provides an explanation of implementing the backpropagation with batch computation.

The work is done by Xingce BAO, Xi FAN and Danjiao MA.

