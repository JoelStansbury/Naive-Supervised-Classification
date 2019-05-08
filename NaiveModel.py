from multiprocessing import Process, Manager, Pool
import numpy as np
import matplotlib.pyplot as plt



def p_y_given_x(x,y,bins,show_hist = False):
    positive_idxs = np.where(y==1)[0]
    negative_idxs = np.where(y==0)[0]
    x_p = x[positive_idxs]
    x_n = x[negative_idxs]
    h_p, b_p = np.histogram(x_p,bins,range=(x.min(),x.max()))
    h_n, b_n = np.histogram(x_n,bins,range=(x.min(),x.max()))
    
    ''' Computationally expensive '''
    h = h_p + h_n + 1
    p_y = np.divide(h_p,h)
    
    ''' Faster, but behaves poorly sometimes '''
    #h_p = h_p/x_p.shape[0]
    #h_n = h_n/x_n.shape[0]
    #p_y = h_p - h_n + 0.5


    if show_hist:
        plt.plot(b_n[1:],p_y)
    return p_y,b_n[1:]



class Naive:
    def __init__(self):
        self.params = []
        self.y = []
        self.num_bins = 20
        
    def fit_1d(self,idx):
        y = self.y
        x = self.x[:,idx]
        num_bins = self.num_bins
        p_y,bins = p_y_given_x(x,y,num_bins)
        return p_y, bins
    
    def predict_1d(self,x,d):
        h,bins = self.params[d]
        return np.interp(x,bins,h)
    
    def fit(self, x, y, b = 20):
        self.y = y
        self.x = x
        self.num_bins = b
        X = list(range(x.shape[1]))
        p = []
        for i in X:
            p.append(self.fit_1d(i))
        self.params = p

    ''' 
    Parallelize the fitting procedure
    
        # this is slower than the for-loop above because we loose the multi-core benifet to matrix operations
        # but, this is how the fitting process should be distributed across a computing cluster
    ------------------------------------
    def fit(self, x, y, b = 20):
        self.y = y
        self.x = x
        self.num_bins = b
        X = list(range(x.shape[1]))
        p = Pool()
        self.params = p.map(self.fit_1d,X)
    ------------------------------------
    '''

    def plotActivations(self):
        params = self.params
        for i in params:
            plt.plot(i[1],i[0])
            plt.show()

    def predict(self, x, give_probs = False):
        
        probs = []
        for i in range(x.shape[1]):
            probs.append(self.predict_1d(x[:,i],i))
        probs = np.array(probs).T
        if give_probs:
            return ((probs.sum(1)/x.shape[1])+1)/2
        return ((probs.sum(1)/x.shape[1])>0.5).astype(int)
            
