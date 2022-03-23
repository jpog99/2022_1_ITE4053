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
K = 5000

#alpha (learning rate)
alpha = 1

w1 = random.random()
w2 = random.random()
b = random.random()

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
for j in range(K):
    W = np.array([w1, w2])
    J = 0
    dw1 = 0
    dw2 = 0
    db = 0
    for i in range(m):
        #foward propagation
        x = np.array([x1_train[i], x2_train[i]])
        z = np.dot(W,x) + b
        a = sigmoid(z)
        J += cross_entropy_loss(a, y_train[i])
        # backward propagation
        dz = a - y_train[i]
        dw1 += x1_train[i] * dz
        dw2 += x2_train[i] * dz
        db += dz
    J /= m
    dw1 /= m
    dw2 /= m
    db /= m

    #update W, b
    w1 -= alpha * dw1
    w2 -= alpha * dw2
    b -= alpha * db

    #print W, b every 500 iteration
    if(j % 500 == 0):
        print(f"[w1, w2, b] = [{w1}, {w2}, {b}]")
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
w_result = np.array([w1, w2])

correct_predict_train = 0
for i in range(m):
    x = np.array([x1_train[i], x2_train[i]])
    z = np.dot(w_result, x)
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
    z = np.dot(w_result, x)
    a = sigmoid(z)
    if(z > 0.5 and y_test[i] == 1):
        correct_predict_test += 1
    elif(z <= 0.5 and y_test[i] == 0):
        correct_predict_test += 1
test_accuracy = correct_predict_test / n * 100
print("Accuracy for 'n' test samples: " + str(test_accuracy) + "%")
