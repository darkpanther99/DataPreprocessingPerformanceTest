import numpy as np
import os
from keras.preprocessing.image import load_img, img_to_array
import time
import concurrent.futures
import threading

'''
Some global constants
'''
PATH = 'train'
IMAGENUM = 25000
THREADNUM = 4
STEP = IMAGENUM / THREADNUM
TARGET_SIZE = (200, 200)
DIM_NUM = 3

'''
A global array for the global test runs
'''
allimgswithrace = np.empty((IMAGENUM, TARGET_SIZE[0], TARGET_SIZE[1], DIM_NUM))

'''
A thread function, which loads the images from the start index to the end index, scales them, and returns them as a numpy array.
'''
def thread_function(start, end):
    #print("szal {} started".format(start))
    threadimgs=[]

    for i, file in enumerate(os.listdir(PATH)):
        if start <= i < end:
            threadimgs.append(img_to_array(load_img(os.path.join(PATH, file), target_size=TARGET_SIZE)))

    threadimgs = np.asarray(threadimgs)
    threadimgs = threadimgs/255.0   #TODO:Move this up to the append line. There may be some performance increase.

    return threadimgs

'''
A thread function to load some images and put them into a globally defined array
'''
def thread_function_with_global_trick(start, end):
    #print("szal {} started".format(start))
    threadimgs=[]

    global allimgswithrace

    for i, file in enumerate(os.listdir(PATH)):
        if start <= i < end:
            allimgswithrace[i] = img_to_array(load_img(os.path.join(PATH, file), target_size=TARGET_SIZE))/255.0

'''
A multithreaded test run, which uses numpy.concatenate, which is slow, because of the array copying
'''
def multithreaded_test():
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i in range(THREADNUM):
            futures.append(executor.submit(thread_function, i * STEP, (i + 1) * STEP))

    allimgs = np.concatenate((futures[0].result(), futures[1].result()))

    for i, future in enumerate(futures):
        if i > 1:
            allimgs = np.concatenate((allimgs, future.result()))

    print(allimgs.shape)


'''
A multithreaded test run, which instead of numpy.concatenate, makes the array in one loop
'''
def multithreaded_test_with_optimized_concatenation():
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i in range(THREADNUM):
            futures.append(executor.submit(thread_function, i * STEP, (i + 1) * STEP))

    allimgs = np.empty((IMAGENUM, TARGET_SIZE[0], TARGET_SIZE[1], DIM_NUM), dtype='float32')

    i=0
    for future in futures:
        for image in future.result():
            allimgs[i] = image
            i+=1

    print(allimgs.shape)

'''
A multithreaded test run, which doesnt concatenate, the results will be in different arrays
This can be helpful if we want separate arrays for separate classes
'''
def multithreaded_test_without_concatenation():
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i in range(THREADNUM):
            futures.append(executor.submit(thread_function, i * STEP, (i + 1) * STEP))

    for i, future in enumerate(futures):
        print(i)
        print(future.result().shape)

'''
A multithreaded test run, which doesnt concatenate, but places the images into a globally defined array
Since the place of the image is predefined, there will be no race conditions, locks are not necessary(no function tries to put two images to the same index)
'''
def multithreaded_test_with_global_trick():
    #TODO:use normal threads instead of threadpool
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i in range(THREADNUM):
            futures.append(executor.submit(thread_function_with_global_trick, i * STEP, (i + 1) * STEP))

    for future in futures:
        print(future.result())


'''
A singlethreaded test run
'''
def singlethreaded_test():

    allimgs=[]

    for i, file in enumerate(os.listdir(PATH)):
        allimgs.append(img_to_array(load_img(os.path.join(PATH, file), target_size=TARGET_SIZE)))

    allimgs=np.asarray(allimgs)
    allimgs=allimgs/255.0

    print(allimgs.shape)



'''
Measuring the time of the test run and printing it out in seconds
'''
start = time.time()
#RUN A TEST HERE!!
multithreaded_test_with_global_trick()
end = time.time()
print("Elapsed time: {} seconds".format(end-start))

