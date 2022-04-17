import numpy as np
import random

# number of train samples
m = 10000

#number of test samples
n = 1000

# train arrays
x1_train = []
x2_train = []
y_train = []

# test arrays
x1_test = []
x2_test = []
y_test = []

# number of iterations
K = 50000

#alpha (learning rate)
alpha = 0.1


# functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cross_entropy_loss(y_hat, y):
    return -(y*np.log10(y_hat + 1e-10) + (1-y)*np.log10(1-y_hat + 1e-10)) #added 1e-10 to avoid NaN problem

def model(x, W, b):
    return sigmoid(np.dot(W, x) + b)

# === STEP 1 ===
#generate m train samples
for i in range(m):
    x1_train.append(random.uniform(-10, 10))
    x2_train.append(random.uniform(-10, 10))
    if x1_train[-1] + x2_train[-1] > 0:
        y_train.append(1)
    else:
        y_train.append(0)

#generate n test samples
for i in range(n):
    x1_test.append(random.uniform(-10, 10))
    x2_test.append(random.uniform(-10, 10))
    if x1_test[-1] + x2_test[-1] > 0:
        y_test.append(1)
    else:
        y_test.append(0)

# === STEP 2 ===
w1 = random.random()
w2 = random.random()
W = np.array([w1, w2])
b = random.random()
for j in range(K):
    x = np.array([x1_train, x2_train])
    Z = np.dot(W,x) + b
    A = sigmoid(Z)

    dZ = A - y_train
    dW = np.dot(x,np.transpose(dZ))/m
    dB = np.sum(dZ)/m

    #update W, b
    W -= alpha * dW
    b -= alpha * dB

    #print W, b every 500 iteration
    if(j % 500 == 0):
        print(f"[w1, w2, b] = [{W[0]}, {W[1]}, {b}]")
    # m cost -> J(W,b)
    x_train = np.array((x1_train[i], x2_train[i]))
    y_hat = model(x_train, W, b)
    cost = np.sum((-cross_entropy_loss(y_hat, y_train[i]))) / m
    if(j % 500 == 0):
        print("Cost: %f" % cost)

# n cost
n_cost = 0
w_result = np.array([w1, w2])
for i in range(n):
    x_test = np.array([x1_test[i], x2_test[i]])
    y_hat = model(x_test, w_result, b)
    n_cost += -cross_entropy_loss(y_hat, y_test[i])
print("Cost with n test samples = ", n_cost / n)

# === STEP 2-4 ===

correct_predict_train = 0
for i in range(m):
    x = np.array([x1_train[i], x2_train[i]])
    z = np.dot(W, x)
    a = sigmoid(z)
    if(z > 0.5 and y_train[i] == 1):
        correct_predict_train += 1
    elif(z <= 0.5 and y_train[i] == 0):
        correct_predict_train += 1
train_accuracy = correct_predict_train / m * 100
print("Accuracy for 'm' train samples: " + str(train_accuracy) + "%")

# === STEP 2-5 ===
correct_predict_test = 0
for i in range(n):
    x = np.array([x1_test[i], x2_test[i]])
    z = np.dot(W, x)
    a = sigmoid(z)
    if(z > 0.5 and y_test[i] == 1):
        correct_predict_test += 1
    elif(z <= 0.5 and y_test[i] == 0):
        correct_predict_test += 1
test_accuracy = correct_predict_test / n * 100
print("Accuracy for 'n' test samples: " + str(test_accuracy) + "%")
