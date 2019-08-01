import numpy as np
import traceback
import sys

def runtest(test,name):
    print('Running Test: %s ... ' % (name),end='')
    try:
        if test():
            print('✔ Passed!')
        else:
            print("✖ Failed!\n The output of your function does not match the expected output. Check your code and try again.")
    except Exception as e:
        print('✖ Failed!\n Your code raises an exception. The following is the traceback of the failure:')
        print(' '.join(traceback.format_tb(sys.exc_info()[2])))


def classify_linear_grader(xs,w,b=None):
    w = w.flatten()    
    predictions=np.zeros(xs.shape[0])
    if b is None:
        b = 0
    predictions = np.sign(xs.dot(w) + b)
    return predictions	