import numpy as np

def block_integration(x_initial,x_final,function,h=0.001):
    x=np.arange(x_initial,x_final+h,h)
    y=function(x)
    return h*np.sum(y[:-1])

def trapezoid_integration(x_initial,x_final,function,h=0.001):
    """
    error O(h^2)
    """
    x=np.arange(x_initial,x_final+h,h)
    y=function(x)
    return h*np.sum(y[1:-1])+h/2*(y[0]+y[-1])

def simpson_integration(x_initial,x_final,function,h=0.001):
    """
    error O(h^3)
    """
    x=np.arange(x_initial,x_final+h,h)
    y=function(x)
    s1=0
    for i in range(1,(len(x)-1)//2+1):
        s1=s1+y[2*i-2]+4*y[2*i-1]+y[2*i]
    return h*s1/3

def monte_carlo_integration(function,density_x,x_function_y,N=10000):
    y=np.random.rand(N)
    x=x_function_y(y)
    s=1/N*np.sum(function(x)/density_x(x))
    return s
def monte_carlo_integration_2d(function,density_x1,density_x2,x1_function_y1,x2_function_y2,N=1000):
    """
    function = lambda x1,x2:f(x1,x2)
    density_x1= lambda x1 :f1(x1)
    density_x2= lambda x2 :f1(x2)
    x1_function_y1 = lambda y1 : x1(y1)
    x2_function_y2 = lambda y2: x1(y2)
    """
    y1=np.random.rand(N)
    y2=np.random.rand(N)
    x1=x1_function_y1(y1)
    x2=x2_function_y2(y2)
    s1=0
    for i in range(N):
        s1=s1+function(x1[i],x2[i])/(density_x1(x1[i])*density_x2(x2[i]))
    return s1/N