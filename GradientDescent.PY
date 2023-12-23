import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def datamaker(n, w, b, err):
    x = np.linspace(-1, 1, n)
    y = (x * w) + b + (err*np.random.randn(n))
    return x, y

x_train , y_train = datamaker(1500, 2.3, -3.8, 0.35)
plt.scatter(x_train, y_train, c="red")
plt.show()

df = pd.DataFrame({"X_Train": x_train, "Y_Train": y_train})
df.head()

x = np.array(df.iloc[:, 0:1])
y = np.array(df.iloc[:, 1:2])



