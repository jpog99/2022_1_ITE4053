import numpy as np
import random
import math

# number of train samples
m = 10000

#number of test samples
n = 1000

# train arrays
x_train = []
y_train = []

# test arrays
x_test = []
y_test = []

# number of iterations
K = 5000

#alpha (learning rate)
alpha = 0.01

# parameters
w1 = random.random()
b1 = random.random()
w2 = random.random()
b2 = random.random()

# functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z + 1e-10))

def cross_entropy_loss(y_hat, y):
    return -(y*np.log10(y_hat + 1e-10) + (1-y)*np.log10(1-y_hat + 1e-10)) #added 1e-10 to avoid NaN problem

def model(x, W, b):
    return sigmoid(np.dot(W, x) + b)

# === STEP 1 ===
#generate m train samples
for i in range(m):
    degree_value = random.uniform(0, 360)
    sine_value = math.sin(math.radians(degree_value))
    x_train.append(degree_value)
    if sine_value > 0:
        y_train.append(1)
    else:
        y_train.append(0)

#generate n test samples
for i in range(n):
    degree_value = random.uniform(0, 360)
    sine_value = math.sin(math.radians(degree_value))
    x_test.append(degree_value)
    if sine_value > 0:
        y_test.append(1)
    else:
        y_test.append(0)

x_train = np.array(x_train).reshape(1, len(x_train))
y_train = np.array(y_train).reshape(1, len(y_train))
x_test = np.array(x_test).reshape(1, len(x_test))
y_test = np.array(y_test).reshape(1, len(y_test))

# === STEP 2 ===
for j in range(K):
    # forward prop
    Z1 = np.dot(w1, x_train) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(w2, A1) + b2
    A2 = sigmoid(Z2)

    # backward prop
    dZ2 = A2 - y_train
    dw2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = np.dot(w2, dZ2) * sigmoid(Z1)
    dW1 = np.dot(dZ1, x_train.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True)

    #update param
    w1 -= alpha * dW1
    b1 -= alpha * db1
    w2 -= alpha * dw2
    b2 -= alpha * db2

    # Step 2-1, 2-2
    #print W, b every 500 iteration
    if(j % 500 == 0):
        print(f"[w1, b1, w2, b2] = [{w1}, {b1}, {w2}, {b2}]")
        cost = np.sum((-cross_entropy_loss(A2, y_train))) / m
        print("Cost: %f" % cost)

# step 2-3
Z1 = model(x_test, w1, b1)
A1 = sigmoid(Z1)
Z2 = model(A1, w2, b2)
A2 = sigmoid(Z2)
n_cost = np.sum((-cross_entropy_loss(A2, y_test))) / n
print("Cost with n test samples = ", n_cost)

# === STEP 2-4 ===
# w1 =-1813.69878086
# w2 = 8.1351926
# b1 = 330901.03284608
# b2 = -4.34741014
correct_predict_train = 0
Z1 = np.dot(w1, x_train) + b1
A1 = sigmoid(Z1)
Z2 = np.dot(w2, A1) + b2
A2 = sigmoid(Z2)
for i in range(m):
    if A2[0,i] > 0.5 and y_train[0,i] == 1:
        correct_predict_train += 1
    elif A2[0,i] < 0.5 and y_train[0,i] == 0:
        correct_predict_train += 1
train_accuracy = correct_predict_train / m * 100
print("Accuracy for 'm' train samples: " + str(train_accuracy) + "%")

# === STEP 2-5 ===
correct_predict_test = 0
Z1 = np.dot(w1, x_test) + b1
A1 = sigmoid(Z1)
Z2 = np.dot(w2, A1) + b2
A2 = sigmoid(Z2)
for i in range(n):
    if A2[0,i] > 0.5 and y_test[0,i] == 1:
        correct_predict_test += 1
    elif A2[0,i] < 0.5 and y_test[0,i] == 0:
        correct_predict_test += 1
test_accuracy = correct_predict_test / n * 100
print("Accuracy for 'n' test samples: " + str(test_accuracy) + "%")