# Optimization

## Specializations - Machine Learning ― Supervised Learning 

## Description

* This repository contains some Optimization exercises
* Please download the following checkpoints for to accompany the following tensorflow main files:

    - graph.ckpt.data-00000-of-00001
    - graph.ckpt.index
    - graph.ckpt.meta


## Learning Objectives

**Understand:**

* What is a hyperparameter?
* How and why do you normalize your input data?
* What is a saddle point?
* What is stochastic gradient descent?
* What is mini-batch gradient descent?
* What is a moving average? How do you implement it?
* What is gradient descent with momentum? How do you implement it?
* What is RMSProp? How do you implement it?
* What is Adam optimization? How do you implement it?
* What is learning rate decay? How do you implement it?
* What is batch normalization? How do you implement it?


## Dependencies
```
Python 3.5
numpy 1.15
tensorflow 1.12
```

## Repo content

* **Main Folder that contains all main of the following tasks:**

| Task | Description |
| --- | --- |
|**0. Normalization Constants** | calculates the normalization (standardization) constants of a matrix
|**1. Normalize** | normalizes (standardizes) a matrix
|**2. Shuffle Data** | shuffles the data points in two matrices the same way
|**3. Mini-Batch** | trains a loaded neural network model using mini-batch gradient descent
|**4. Moving Average** | calculates the weighted moving average of a data set
|**5. Momentum** | updates a variable using the gradient descent with momentum optimization algorithm
|**6. Momentum Upgraded** | creates the training operation for a neural network in tensorflow using the gradient descent with momentum optimization algorithm
|**7. RMSProp** | that updates a variable using the RMSProp optimization algorithm
|**8. RMSProp Upgraded** | creates the training operation for a neural network in tensorflow using the RMSProp optimization algorithm
|**9. Adam** | updates a variable in place using the Adam optimization algorithm
|**10. Adam Upgraded** | creates the training operation for a neural network in tensorflow using the Adam optimization algorithm
|**11. Learning Rate Decay** | updates the learning rate using inverse time decay in numpy
|**12. Learning Rate Decay Upgraded** | creates a learning rate decay operation in tensorflow using inverse time decay
|**13. Batch Normalization** | normalizes an unactivated output of a neural network using batch normalization
|**14. Batch Normalization Upgraded** | creates a batch normalization layer for a neural network in tensorflow
|**15. Put it all together and what do you get?** | that builds, trains, and saves a neural network model in tensorflow using Adam optimization, mini-batch gradient descent, learning rate decay, and batch normalization

## Usage
* Clone the repo and execute the main files

## Author
- [Cristian G](https://github.com/cristian-fg)

## License
[MIT](https://choosealicense.com/licenses/mit/)
