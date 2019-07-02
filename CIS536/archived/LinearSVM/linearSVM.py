#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().magic('load_ext autoreload')


# In[2]:


# get_ipython().magic('autoreload 2')
import numpy as np
from numpy.matlib import repmat
import sys
import time

import helper as h

import l2distance
import visclassifier

import matplotlib
import matplotlib.pyplot as plt


from scipy.stats import linregress

import pylab
from matplotlib.animation import FuncAnimation

# get_ipython().magic('matplotlib notebook')


# <!--announcements-->
# <blockquote>
#     <center>
#     <img src="yinyang.png" width="400px" /></a>
#     </center>
#       <p><cite><center>"Just as we have two eyes and two feet,<br>
#       duality is part of life."<br>
# <b>--Carlos Santana</b><br>
#       </center></cite></p>
# </blockquote>
# 
# 
# <h3>Introduction</h3>
# In this project, you will implement a linear support vector machine. To start, we will generate a linear separable dataset and visualize it.

# In[3]:


xTr, yTr = h.generate_data()
h.visualize_2D(xTr, yTr)


# <p>Recall that the unconstrained loss function for linear SVM is 
#     
# $$
# \begin{aligned}
# \min_{\mathbf{w},b}\underbrace{\mathbf{w}^T\mathbf{w}}_{l_{2}-regularizer} +  C\  \sum_{i=1}^{n}\underbrace{\max\left [ 1-y_{i}(\mathbf{w}^T \mathbf{x}_i+b),0 \right ]}_{hinge-loss}
# \end{aligned}
# $$
# You will need to implement the function <code>hinge</code>, which takes in training data <code>xTr</code> ($n\times d$) and labels <code>yTr</code> ($n$) with <code>yTr[i]</code>$\in \{-1,1\}$ and evaluate the loss of classifier $(\mathbf{w},b)$. In addition, you need to implement <code>hinge_grad</code> that takes in the same arguments but return gradient of the loss function with respect to $(\mathbf{w},b)$. </p>

# In[4]:


def hinge(w, b, xTr, yTr, C):
    """
    INPUT:
    w     : d   dimensional weight vector
    b     : scalar (bias)
    xTr   : nxd dimensional matrix (each row is an input vector)
    yTr   : n   dimensional vector (each entry is a label)
    C     : scalar (constant that controls the tradeoff between l2-regularizer and hinge-loss)
    
    OUTPUTS:
    loss     : the total loss obtained with (w, b) on xTr and yTr (scalar)
    """
    
    loss = 0.0
    ### BEGIN SOLUTION
    margin = yTr*(xTr @ w + b)
    
#     # Need to avoid numerical issue
#     margin[np.isclose(margin, 1)] = 1
    
#     loss = w.T @ w + C*(np.sum(np.maximum(1 - margin, 0)))
    loss = w.T @ w + C*(np.sum(np.maximum(1 - margin, 0) ** 2))
    ### END SOLUTION
    
    return loss


# In[5]:


def hinge_grad(w, b, xTr, yTr, C):
    """
    INPUT:
    w     : d   dimensional weight vector
    xTr   : nxd dimensional matrix (each row is an input vector)
    yTr   : n   dimensional vector (each entry is a label)
    C     : constant (scalar that controls the tradeoff between l2-regularizer and hinge-loss)
    
    OUTPUTS:
    wgrad :  d dimensional vector (the gradient of the hinge loss with respect to the weight, w)
    bgrad :  constant (the gradient of he hinge loss with respect to the bias, b)
    """
    n, d = xTr.shape
    
    wgrad = np.zeros(n)
    bgrad = np.zeros(1)
    
    ### BEGIN SOLUTION
    margin = yTr*(xTr @ w + b)
    
    hinge = np.maximum(1 - margin, 0)
    
    indicator = (1 - margin > 0).astype(int)
    wgrad = 2 * w + C * np.sum((2 * hinge * indicator * -yTr).reshape(-1, 1) * xTr, axis=0)
    bgrad = C * np.sum(2 * hinge * indicator * -yTr, axis=0)
    
    ### END SOLUTION
    return wgrad, bgrad


# By calling the following minimization routine we implement for you, you will obtain your linear SVM. 

# In[6]:


w, b, final_loss = h.minimize(objective=hinge, grad=hinge_grad, xTr=xTr, yTr=yTr, C=1000)
print('The Final Loss of your model is: {:0.4f}'.format(final_loss))


# Now, let's visualize the decision boundary on our linearly separable dataset. Since the dataset is linearly separable,  you should obtain $0\%$ training error with sufficiently large values of $C$ (e.g. $C>1000$). 

# In[7]:


h.visualize_classfier(xTr, yTr, w, b)

# Calculate the training error
err=np.mean(np.sign(xTr.dot(w) + b)!=yTr)
print("Training error: {:.2f} %".format (err*100))


# In[9]:


Xdata = []
ldata = []

fig = plt.figure()
details = {
    'w': None,
    'b': None,
    'stepsize': 1,
    'ax': fig.add_subplot(111), 
    'line': None
}

plt.xlim(0,1)
plt.ylim(0,1)
plt.title('Click to add positive point and shift+click to add negative points.')

def updateboundary(Xdata, ldata, details):
    w_pre, b_pre, _ = h.minimize(objective=hinge, grad=hinge_grad, xTr=np.concatenate(Xdata), 
            yTr=np.array(ldata), C=1000)
    details['w'] = np.array(w_pre).reshape(-1)
    details['b'] = b_pre
    details['stepsize'] += 1

def updatescreen(details):
    b = details['b']
    w = details['w']
    q = -b / (w**2).sum() * w
    if details['line'] is None:
        details['line'], = details['ax'].plot([q[0] - w[1],q[0] + w[1]],[q[1] + w[0],q[1] - w[0]],'b--')
    else:
        details['line'].set_ydata([q[1] + w[0],q[1] - w[0]])
        details['line'].set_xdata([q[0] - w[1],q[0] + w[1]])

def generate_animate(Xdata, ldata, details):
    def animate(i):
        if (len(ldata) > 0) and (np.amin(ldata) + np.amax(ldata) == 0):
            if details['stepsize'] < 1000:
                updateboundary(Xdata, ldata, details)
                updatescreen(details)
    return animate

def generate_onclick(Xdata, ldata):    
    def onclick(event):
        if event.key == 'shift': 
            # add positive point
            details['ax'].plot(event.xdata,event.ydata,'or')
            label = 1
        else: # add negative point
            details['ax'].plot(event.xdata,event.ydata,'ob')
            label = -1    
        pos = np.array([[event.xdata, event.ydata]])
        ldata.append(label)
        Xdata.append(pos)
    return onclick


cid = fig.canvas.mpl_connect('button_press_event', generate_onclick(Xdata, ldata))
ani = FuncAnimation(fig, generate_animate(Xdata, ldata, details), np.arange(1,100,1), interval=10)
plt.show()


# In[ ]:




