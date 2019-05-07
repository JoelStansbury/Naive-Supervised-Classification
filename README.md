# Naive-Supervised-Classification
Easily parallelizable supervised classification algorithm



The IPython Notebook in the repository gives a breif explanation of the basis of this model as well as showing some examples where the model performs well and not so well. In summary, it scales very well with number of datapoints. The notebook shows a model trained using 1,000,000 X 2 feature matrix in 0.06 seconds with an accuracy of 98.2, Compared to SKLearn Gaussian Naive Bayes which acheived an accuracy of 99.1 in 0.18 seconds. That said, the algorithm is called Naive not only because it treats all dimensions as independent, but also because the model is baised on some of the most hand-wavy math I've ever come up with. Because of this, there is some strange behaviour, particularly when there is not great separation between the clusters. Finally, the strange behaviour seems to become less of a problem when there are more dimensions, but the speed advantage over SKLearn's Naive Bayes model is lost.


```
class Naive:
    def __init__(self):
        self.params = []
    def fit_1d(self,x,y,bins = 20, show_hist = False):
        positive_idxs = np.where(y==1)[0]
        negative_idxs = np.where(y==0)[0]
        x_p = x[positive_idxs]
        x_n = x[negative_idxs]
        h_p, b_p = np.histogram(x_p,bins,range=(x.min(),x.max()))
        h_p = h_p/x_p.shape[0]
        h_n, b_n = np.histogram(x_n,bins,range=(x.min(),x.max()))
        h_n = h_n/x_n.shape[0]
        h = h_p - h_n
        if show_hist:
            plt.plot(b_n[1:],h)
        return h,b_n[1:]
    
    def predict_1d(self,x,d):
        h,bins = self.params[d]
        return np.interp(x,bins,h)
    
    def fit(self, x, y, b = 20):
        for i in range(x.shape[1]):
            self.params.append(self.fit_1d(x[:,i], y, b))
    def predict(self, x, give_probs = False):
        probs = []
        for i in range(x.shape[1]):
            probs.append(self.predict_1d(x[:,i],i))
        probs = np.array(probs).T
        if give_probs:
            return ((probs.sum(1)/x.shape[1])+1)/2
        return ((probs.sum(1)/x.shape[1])>0).astype(int)
            
```
