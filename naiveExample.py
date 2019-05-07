import numpy as np
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from time import time
from NaiveModel import Naive

def shuffle(x,y):
    z = list(zip(x,y))
    np.random.shuffle(z)
    x,y = zip(*z)
    return np.array(x),np.array(y)

def split(x,y,split_percentage):
    split_idx = int(len(y)*split_percentage)
    z = list(zip(x,y))
    z1 = z[:split_idx]
    z2 = z[split_idx:]
    x1,y1 = zip(*z1)
    x2,y2 = zip(*z2)
    return np.array(x1),np.array(y1),np.array(x2),np.array(y2)




'''
Use the x_value of all training points to generate p(y|x)
'''
def p_y_given_x(x,y,bins,show_hist = False):
    positive_idxs = np.where(y==1)[0]
    negative_idxs = np.where(y==0)[0]
    x_p = x[positive_idxs]
    x_n = x[negative_idxs]
    h_p, b_p = np.histogram(x_p,bins,range=(x.min(),x.max()))
    h_p = h_p/x_p.shape[0]
    h_n, b_n = np.histogram(x_n,bins,range=(x.min(),x.max()))
    h_n = h_n/x_n.shape[0]
    h = h_p + h_n
    p_y = np.divide(h_p,h)
    if show_hist:
        plt.plot(b_n[1:],p_y)
    return p_y,b_n[1:]


def gen_random(p_size, n_size, dims):
    p_mean = np.random.randint(50, size = dims)
    n_mean = np.random.randint(50, size = dims)
    p_sig = (np.random.randint(5, size = dims)+1)*10
    n_sig = (np.random.randint(5, size = dims)+1)*10

    p_data = np.random.normal(p_mean, p_sig, (p_size,dims))
    n_data = np.random.normal(n_mean, n_sig, (n_size,dims))
    labels = np.array([1]*p_size+[0]*n_size)

    data = np.vstack((p_data,n_data))
    data,labels = shuffle(data,labels)
    return data,labels

def evaluate(model, X, y):
    X,y = shuffle(X,y)
    x_train, y_train, x_test, y_actual = split(X,y,.70)
    model = model()
    t0 = time()
    model.fit(x_train,y_train)
    t1 = time()

    try:
        model.plotActivations()
    except:
        pass
    y_predict = model.predict(x_test)
    t2 = time()
    correct = y_predict == y_actual
    accuracy = correct.sum()/correct.shape[0]
    fp = (y_predict-1 == y_actual).sum()/(y_predict.sum())
    tn = (y_predict+1 == y_actual).sum()/((y_predict-1).sum()*-1)
    print("Accuracy: {}\nFalse Positives: {}\nTrue Negatives: {}\nTime to fit: {}\nTime to predict: {}".format(accuracy,fp, tn, t1-t0, t2-t1))
    plt.scatter(x_test[:,0],x_test[:,1],cmap = 'cividis', c = y_predict, marker = '.')
    plt.show()
    

X,y = gen_random(10000, 10000, 2)
#plt.scatter(data[:,0],data[:,1],cmap = 'cividis', c = labels, marker = '.')
#plt.show()

print("\n\nModel: Naive\n-----------------------")
evaluate(Naive, X,y)

print("\n\nModel: GaussianNB\n-----------------------")
evaluate(GaussianNB, X,y)




'''
-------------------------------------------------------------------
----------Test on IRIS Dataset-------------------------------------
-------------------------------------------------------------------
'''


# import some data to play with
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data  # we only take the first two features.
y = (np.array(iris.target) ==0).astype(int)

print("\n\nIRIS\nModel: Naive\n-----------------------")
evaluate(Naive, X,y)

print("\n\nModel: GaussianNB\n-----------------------")
evaluate(GaussianNB, X,y)



'''
-------------------------------------------------------------------
----------Test on DIGITS Dataset-------------------------------------
-------------------------------------------------------------------
'''


# import some data to play with
from sklearn import datasets
digits = datasets.load_digits()
X = digits.data  # we only take the first two features.
y = (np.array(digits.target) ==0).astype(int)


print("\n\nDIGITS\nModel: Naive\n-----------------------")
evaluate(Naive, X,y)

print("\n\nModel: GaussianNB\n-----------------------")
evaluate(GaussianNB, X,y)




