import numpy as np

# activation functions
def sigmoid(x):
    return 1/ (1 + np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def step_function(x):
    y = x > 0 
    return y.astype(int)

def relu(x): 
    return np.maximum(0,x)

# identity fucntion
def softmax(a): 
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y