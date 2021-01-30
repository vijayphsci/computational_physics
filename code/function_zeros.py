import numpy as np
import matplotlib.pyplot as plt
import derivatives as der
def bisection_auto(lower,upper,function,epsilon=1e-6,max_iteration=100000):
    """
    condition
    |f(xn)|<epsilon
    xn=approximation of zero of function
    """
    temp=2*epsilon
    itr=0
    while abs(temp)>epsilon:
        c=(upper+lower)/2
        temp=function(c)
        if temp==0:
            return c
        elif temp*function(upper)<0:
            if function(lower)==0:
                return lower
            lower=c
        else:
            if function(upper)==0:
                return upper
            upper=c
        if itr>max_iteration:
            print('bisection auto method may not converge')
            return c
        itr=itr+1
    return c
def bisection_itr(lower,upper,function,delta=1e-6):
    """
    condition
    |xn-s|<delta
    s=true zero of function
    xn=approximation of zero of function
    """
    max_itr=int(np.log(abs(upper-lower)/(2*delta))/np.log(2))+1
    for i in range(max_itr):
        c=(upper+lower)/2
        temp=function(c)
        if temp==0:
            return c
        elif temp*function(upper)<0:
            if function(lower)==0:
                return lower
            lower=c
        else:
            if function(upper)==0:
                return upper
            upper=c
    return c
def newton_rapshon(start,function,epsilon=1e-6,h=0.001,max_iteration=10000):
    """
    condition
    |f(xn)|<epsilon
    xn=approximation of zero of function
    
    """
    temp=2*epsilon
    x=start
    itr=0
    if function(x)==0:
        return x
    while abs(temp)>epsilon:
        x=x-function(x)/der.derivative_3p(x,function,h)
        temp=function(x)
        if itr>max_iteration:
            print('newton rapshon method may not converging')
            return x
        itr=itr+1

    return x

def newton_rapshon_2d(x1_start,x2_start,function1,function2,epsilon=1e-6,h=0.001,max_iteration=100000):
    """
    function1 = lambda x1,x2:f1(x1,x2)
    function2 = lambda x1,x2:f2(x1,x2)
    max_iteration = maximum iteration to run for newton rapshon if exceed then stop method may not converge
    """
    temp1,temp2=2*epsilon,2*epsilon
    x1=x1_start
    x2=x2_start
    if function1(x1,x2)==0 and function1(x1,x2)==0:
        return x1,x2
    x=np.array([x1,x2]).reshape(2,1)
    f=np.zeros((2,1))
    empty=np.zeros((2,2))
    itr=0
    while (abs(temp1)>epsilon and abs(temp2)>epsilon):
        f[0,0],f[1,0]=function1(x[0,0],x[1,0]),function2(x[0,0],x[1,0])
        empty[0,0],empty[0,1]=der.partial_derivative_3p(x[0,0],x[1,0],function1,h)
        empty[1,0],empty[1,1]=der.partial_derivative_3p(x[0,0],x[1,0],function2,h)
        x=x-np.dot(np.linalg.inv(empty),f)
        temp1,temp2=function1(x[0,0],x[1,0]),function2(x[0,0],x[1,0])
        if itr>max_iteration:
            print('newton rapshon 2d method may not converging')
            return x[0,0],x[1,0]
        itr=itr+1
    return x[0,0],x[1,0]

def extremum(lower,upper,function,epsilon=1e-6,h=0.001,method='newton_rapshon',max_iteration=100000,step=2,kind=False):
    """
    function = lambda x:f(x)
    method = newton_rapshon,bisection_itr,bisection_auto
    lower=lower value for bisection method, start value for newton rapshon
    upper=upper value for bisection method , give any random value for newton rapshon
    max_iteration = maximum iteration to run for newton rapshon or bisection auto if exceed then stop method may not converge
    step = number of step taken to check extremum
    return extremum point , kind of extremum  :: if kind = True
    return extremum point :: if kind =False
    """
    if method=='newton_rapshon':
        ext=newton_rapshon(lower,lambda x:der.derivative_3p(x,function,h),epsilon,h,max_iteration)
    elif method=='bisection_auto':
        ext=bisection_auto(lower,upper,lambda x:der.derivative_3p(x,function,h),epsilon,max_iteration)
    elif method=='bisection_itr':
        ext=bisection_itr(lower,upper,lambda x:der.derivative_3p(x,function,h),epsilon)
    t1=function(ext-h*step)
    t2=function(ext)
    t3=function(ext+h*step)
    if t1>t2 and t3>t2:
        kind = 'minima'
    elif t2>t1 and t2>t3:
        kind = 'maxima'
    else:
        kind = 'saddle'
    if kind:
        return ext,kind
    else:
        return ext




