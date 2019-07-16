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


def ridge_grader(w,xTr,yTr,lmbda):
    n, d = xTr.shape
    preds = xTr @ (w)
    diff = preds - yTr
    loss = np.mean(diff ** 2) + lmbda * w.dot(w)
    grad = 2.0 * (xTr.T) @(diff) /n  + 2*lmbda * w
    
    return loss, grad


def grad_descent_grader(func,w,alpha,maxiter,delta=1e-02):
    losses = np.zeros(maxiter)
    for i in range(maxiter):
        losses[i], grad = func(w)
        w -= alpha * grad
        if np.linalg.norm(grad) < delta: 
            losses = losses[:i+1]
            break
    return w, losses


def logistic_grader(w,xTr,yTr, lmbda):
    n, d = xTr.shape
    
    e = np.exp((xTr@w).flatten() * - yTr)
    loss = np.sum(np.log(1+e)) / n + lmbda * w.dot(w)
    grad = np.dot(e * -yTr/(1+e), xTr) / n + 2*lmbda * w
    
    return loss,grad


def linclassify_grader(w,xTr):
    return np.sign(xTr.dot(w))


def analyze_grader(kind,truth,preds):
    truth = truth.flatten()
    preds = preds.flatten()
    if kind == 'abs':
        # compute the absolute difference between truth and predictions
        output = np.sum(np.abs(truth - preds)) / float(len(truth))
    elif kind == 'acc':
        if len(truth) == 0 and len(preds) == 0:
            output = 0
        else:
            output = np.sum(truth == preds) / float(len(truth))
    return output
    