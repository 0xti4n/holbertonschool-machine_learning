# Hyperparameter Tuning

## Specializations - Machine Learning ― Unsupervised Learning

## Description

* This repository contains Hyperparameter Tuning exercises

## Learning Objectives

**Understand:**

* What is Hyperparameter Tuning?
* What is random search? grid search?
* What is a Gaussian Process?
* What is a mean function?
* What is a Kernel function?
* What is Gaussian Process Regression/Kriging?
* What is Bayesian Optimization?
* What is an Acquisition function?
* What is Expected Improvement?
* What is Knowledge Gradient?
* What is Entropy Search/Predictive Entropy Search?
* What is GPy?
* What is GPyOpt?

## Dependencies
```
Python 3.5
numpy 1.15
GPyOpt
```

```python
pip install --user GPy
pip install --user gpyopt
```

## Repo content

* **Main Folder that contains all main of the following tasks:**

| Task | Description |
| --- | --- |
|**0. Initialize Gaussian Process**| class GaussianProcess that represents a noiseless 1D Gaussian process
|**1. Gaussian Process Prediction**| predicts the mean and standard deviation of points in a Gaussian process
|**2. Update Gaussian Process**| updates a Gaussian Process
|**3. Initialize Bayesian Optimization**| class BayesianOptimization that performs Bayesian optimization on a noiseless 1D Gaussian process
|**4. Bayesian Optimization - Acquisition**| calculates the next best sample location
|**5. Bayesian Optimization**| optimizes the black-box function

## Usage
* Clone the repo and execute the main files

## Author
- [Cristian G](https://github.com/cristian-fg)

## License
[MIT](https://choosealicense.com/licenses/mit/)
