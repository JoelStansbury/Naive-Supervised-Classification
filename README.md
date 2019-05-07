# Naive-Supervised-Classification
Easily parallelizable supervised classification algorithm



The IPython Notebook in the repository gives a breif explanation of the basis of this model as well as showing some examples where the model performs well and not so well. In summary, it scales very well with number of datapoints. The notebook shows a model trained using 1,000,000 X 2 feature matrix in 0.06 seconds with an accuracy of 98.2, Compared to SKLearn Gaussian Naive Bayes which acheived an accuracy of 99.1 in 0.18 seconds. That said, the algorithm is called Naive not only because it treats all dimensions as independent, but also because the model is baised on some of the most hand-wavy math I've ever come up with. Because of this, there is some strange behaviour, particularly when there is not great separation between the clusters. Finally, the strange behaviour seems to become less of a problem when there are more dimensions, but the speed advantage over SKLearn's Naive Bayes model is lost.
