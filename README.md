# DataPreprocessingPerformanceTest
A python script to measure the speed of image preprocessing for Deep Learning

Working with large datasets for Deep Learning applications is a slow process.  
Not only the training takes a lot of time, even on GPU, preprocessing the data can be slow as well.  
Generally the advised thing to do is to use a <a href = "https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator"> data generator</a>, to generate data on the fly, while training the model, but I wanted to do some tests to measure the speed of data preprocessing before the training.

The task is an easy one: Load the images of the <a href = "https://www.kaggle.com/c/dogs-vs-cats"> Kaggle Cats vs Dogs </a> dataset into a numpy array, and minmax scale them to be ready for training.  

My first idea was to use multithreading to speed up the process, since converting the images to RGB pixel numbers, and scaling them runs on the CPU, but later I discovered some additional performance improving ideas.
