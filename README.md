# Statistics for Machine Learning
This is the code repository for [Statistics for Machine Learning](https://www.packtpub.com/big-data-and-business-intelligence/statistics-machine-learning?utm_source=github&utm_medium=repository&utm_campaign=9781788295758), published by [Packt](https://www.packtpub.com/?utm_source=github). It contains all the supporting project files necessary to work through the book from start to finish.
## About the Book
Complex statistics in Machine Learning worry a lot of developers. Knowing statistics helps you build strong Machine Learning models that are optimized for a given problem statement. This book will teach you all it takes to perform complex statistical computations required for Machine Learning. You will gain information on statistics behind supervised learning, unsupervised learning, reinforcement learning, and more. You will see real-world examples that discuss the statistical side of Machine Learning and familiarize yourself with it. You will come across programs for performing tasks such as model, parameter fitting, regression, classification, density collection, working with vectors, matrices, and more. By the end of the book, you will have mastered the required statistics for Machine Learning and will be able to apply your new skills to any sort of industry problem.
## Instructions and Navigation
All of the code is organized into folders. Each folder starts with a number followed by the application name. For example, Chapter02.



The code will look like the following:
```
>>> import numpy as np
>>> from scipy import stats
>>> data = np.array([4,5,1,2,7,2,6,9,3])
# Calculate Mean
>>> dt_mean = np.mean(data) ; print ("Mean :",round(dt_mean,2))
# Calculate Median
>>> dt_median = np.median(data) ; print ("Median :",dt_median)
# Calculate Mode
>>> dt_mode = stats.mode(data); print ("Mode :",dt_mode[0][0])
```

This book assumes that you know the basics of Python and R and how to install the
libraries. It does not assume that you are already equipped with the knowledge of advanced
statistics and mathematics, like linear algebra and so on.
The following versions of software are used throughout this book, but it should run fine
with any more recent ones as well:
* Anaconda 3–4.3.1 (all Python and its relevant packages are included in
Anaconda, Python 3.6.1, NumPy 1.12.1, Pandas 0.19.2, and scikit-learn 0.18.1)
* R 3.4.0 and RStudio 1.0.143
* Theano 0.9.0
* Keras 2.0.2

## Related Products
* [Machine Learning for Developers](https://www.packtpub.com/big-data-and-business-intelligence/machine-learning-developers?utm_source=github&utm_medium=repository&utm_campaign=9781786469878)

* [Scala for Machine Learning](https://www.packtpub.com/big-data-and-business-intelligence/scala-machine-learning?utm_source=github&utm_medium=repository&utm_campaign=9781783558742)

* [Machine Learning for OpenCV](https://www.packtpub.com/big-data-and-business-intelligence/machine-learning-opencv?utm_source=github&utm_medium=repository&utm_campaign=9781783980284)
