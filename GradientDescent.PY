import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

w, b, n = 2.3, -3.8, 1500
def datamaker(n, w, b, err):
    x = np.linspace(-1, 1, n)
    y = (x * w) + b + (err*np.random.randn(n))
    return x, y

x_train , y_train = datamaker(n, w, b, 0.35)
plt.scatter(x_train, y_train, c="red")
plt.show()

df = pd.DataFrame({"X_Train": x_train, "Y_Train": y_train})
df.head()

X = np.array(df.iloc[:, 0:1])
y = np.array(df.iloc[:, 1:2])


def costfunc(wl, bl, X, y):
    z = np.dot(X, wl) + bl
    loss = (z-y)**2
    m = len(z)
    j = np.sum(loss)/m
    return z, j

def gradient(wl, bl, X, y):
    z = np.dot(X, wl) + bl
    m = X.shape[0]
    dw = (-2/m)*np.dot(X.T, (y-z))
    db = (-2/m)*np.sum(y-z)
    return dw, db

#Training
wl = np.random.randint(0, 10)
bl = np.random.randint(0, 10)

y_pred = wl*X + bl
plt.scatter(x_train, y_train)
plt.scatter(X, y_pred)
plt.show()

def learning(wl, bl, X, y, iteration, grate):
    z, j = costfunc(wl, bl, X, y)
    costs = []
    costs.append(j)

    for i in range(iteration):
        dw, db = gradient(wl, bl, X, y)
        wl = wl - (grate * dw)
        bl = bl - (grate * db)
        costs.append(costfunc(wl, bl, X, y)[1])
    wf = wl
    bf = bl

    return wf, bf, costs

wf, bf, costs = learning(wl, bl, X, y, 10000, 0.005)
yf = wf*x_train+bf

plt.figure()
plt.plot(costs)
plt.xlabel("Iterations")
plt.ylabel("Error")
plt.title("Optimization Progress")
plt.grid()
plt.show()

plt.figure()
plt.scatter(x_train, y_train, c="blue")
plt.plot(x_train, wf[0][0]*x_train+bf, c="red", lw=4)
plt.grid()
plt.show()

