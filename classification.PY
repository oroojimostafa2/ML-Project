import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

n_samples = 1000

phi_s = np.abs(np.random.normal(0.25, 0.01, n_samples)) 
swc_s = np.abs(np.random.normal(0.1, 0.01, n_samples)) 
k_s =  8.58102 * (phi_s**(4.4)) / (swc_s**2)

phi_l = np.abs(np.random.normal(0.215, 0.01, n_samples)) 
swc_l = np.abs(np.random.normal(0.1, 0.01, n_samples)) 
k_l = 8.58102 * (phi_s**(4.4)) / (swc_s**2)

#DataFrame Sandstone
df_s = pd.DataFrame({'porosity':phi_s, 'swc':swc_s, 'permeability': k_s})
df_s['label'] = 1
label_s = np.array(df_s["label"])
df_s["lithology"] = "sandstone"
litholgy_s = np.array(df_s["lithology"])

#DataFrame Limestone
df_l = pd.DataFrame({'porosity':phi_l, 'swc':swc_l, 'permeability': k_l})
df_l['label'] = 0
label_l = np.array(df_l["label"])
df_l["lithology"] = "limestone"
litholgy_l = np.array(df_l["lithology"])

# Combinig and Shuffleing
df = pd.merge(df_l, df_s, how="outer")
df = df.sample(frac=1, ignore_index=True)

# Let us Start Training
X = np.array([df["porosity"], df["swc"], df["permeability"]])
y = np.array([df["label"]])

def sigma(z):
    return 1/(1+np.exp(-z))

def initialize(X):
    #
    #To initialize the weights and bias.
    #Here two features so nx = 2
    #Hence Weight matrix must be (2,1)

    w = np.zeros((X.shape[0], 1))
    b = 0
    return w, b

def fpropagate(X, w, b):
    # w - (2*1)
    # X = (2*m)
    # Z = w'.X + b ==> (1*2).(2,m) + 0 ==> (1,m)

    Z = np.dot(w.T, X) + b
    A = sigma(Z)
    return A

def cost(X, y, w, b):

    A = fpropagate(X, w, b)
    L = y*np.log(A) + (1-y)*np.log(1-A)
    m = X.shape[1]
    J = (-1/m) * np.sum(L)

    return J

def gradient(X, y, w, b):

    m = X.shape[1]
    A = fpropagate(X, w, b)
    dw = (1/m)*np.dot(X, (A-y).T)
    db = (1/m)*np.sum(A-y)

    return dw, db

def backprop(X, y, w, b, epoch, learning_rate):

    
    for i in range(epoch):
        dw, db = gradient(X, y, w, b)
        w = w -  (learning_rate*dw)
        b = b - (learning_rate*db)
    
    return w, b


def predict(X, w, b):
    yp = np.zeros((1, X.shape[1]))
    Z = np.dot(w.T, X) + b
    A = sigma(Z)

    for i in range(X.shape[1]):
        if A[0, i] >= 0.5:
            yp[0, i] = 1
        else:
            yp[0, i] = 0
    return yp 

def report(y, yp):
    
    df_r = pd.DataFrame({"label": y.reshape(y.shape[1]), "prediction": yp.reshape(yp.shape[1])})
    df_r['minus'] = df_r["label"] - df_r["prediction"]
    df_r["tf"] = 0
    for i in range(df_r.shape[0]):
        if df_r["minus"].iloc[i] == -1:
            df_r["tf"].iloc[i] = "FP"
        elif df_r["minus"].iloc[i] == 1:
            df_r["tf"].iloc[i] = "FN"
        elif df_r["minus"].iloc[i] == 0:
            if df_r["prediction"].iloc[i] == 0:
                df_r["tf"].iloc[i] = "TN"
            else:
                df_r["tf"].iloc[i] = "TP"
    TP = df_r[df_r["tf"]=="TP"].shape[0]
    TN = df_r[df_r["tf"]=="TN"].shape[0]
    FP = df_r[df_r["tf"]=="FP"].shape[0]
    FN = df_r[df_r["tf"]=="FN"].shape[0]

    df_report = pd.DataFrame({"TP": [TP], "TN": [TN], "FP": [FP], "FN": [FN],\
                               "percisionT": [TP/(TP+FP)], "recallT": [TP/(TP+FN)]\
                                , "percisionF": [TN/(TN+FN)], "recallF": [TN/(TN+FP)]})

    return df_report

w, b = initialize(X)
wf, bf = backprop(X, y, w, b, 10000, 0.05)
yp = predict(X, wf, bf)
df_report = report(y, yp)

# Now let us use the scikit model
model = LogisticRegression()
model.fit(X.T, y.reshape(2*n_samples))
print(classification_report(y, yp))
