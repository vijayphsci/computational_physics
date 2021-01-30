import matplotlib.pyplot as plt
import numpy as np
def dx_dt_euler(t_initial,t_final,x_initial,dxdt,dt=0.01):
    """
    Eulers method
    dxdt = lambda x,t : f(x,t)
    error h=dt ,O(h^2)
    return np array x,t
    """
    n=int((t_final-t_initial)/dt)
    if n<0:
        n=-n
        dt=-dt
    x=np.zeros(n+1)
    t=np.zeros(n+1)
    t[0]=t_initial
    x[0]=x_initial
    for i in range(n):
        x[i+1]=x[i]+dt*dxdt(x[i],t[i])
        t[i+1]=t[i]+dt
    return x,t

def dx_dt_predictor_corrector(t_initial,t_final,x_initial,dxdt,dt=0.01):
    """
    predictor corrector method
    dxdt = lambda x,t : f(x,t)
    error h=dt ,O(h^2)
    return np array x,t
    """
    n=int((t_final-t_initial)/dt)
    if n<0:
        n=-n
        dt=-dt
    x=np.zeros(n+1)
    t=np.zeros(n+1)
    t[0]=t_initial
    x[0]=x_initial
    for i in range(n):
        t[i+1]=t[i]+dt
        euler=x[i]+dt*dxdt(x[i],t[i])
        x[i+1]=x[i]+dt/2*(dxdt(x[i],t[i])+dxdt(euler,t[i+1]))
    return x,t

def dx_dt_rk2(t_initial,t_final,x_initial,dxdt,dt=0.01):
    """
    RK2 method
    dxdt = lambda x,t : f(x,t)
    error h=dt ,O(h^3)
    return np array x,t
    """    
    n=int((t_final-t_initial)/dt)
    if n<0:
        n=-n
        dt=-dt
    x=np.zeros(n+1)
    t=np.zeros(n+1)
    t[0]=t_initial
    x[0]=x_initial
    for i in range(n):
        t[i+1]=t[i]+dt
        k1=dt*dxdt(x[i],t[i])
        k2=dt*dxdt(x[i]+k1/2,t[i]+dt/2)
        x[i+1]=x[i]+k2
    return x,t

def dx_dt_rk4(t_initial,t_final,x_initial,dxdt,dt=0.01):
    """
    RK2 method
    dxdt = lambda x,t : f(x,t)
    error h=dt ,O(h^5)
    return np array x,t
    """
    n=int((t_final-t_initial)/dt)
    if n<0:
        n=-n
        dt=-dt
    x=np.zeros(n+1)
    t=np.zeros(n+1)
    t[0]=t_initial
    x[0]=x_initial
    for i in range(n):
        k1=dt*dxdt(x[i],t[i])
        k2=dt*dxdt(x[i]+k1/2,t[i]+dt/2)
        k3=dt*dxdt(x[i]+k2/2,t[i]+dt/2)
        k4=dt*dxdt(x[i]+k3,t[i]+dt)
        x[i+1]=x[i]+k1/6+k2/3+k3/3+k4/6
        t[i+1]=t[i]+dt
    return x,t

def dx1_dt_dx2_dt_rk4(t_initial,t_final,x1_initial,x2_initial,dx1dt,dx2dt,dt=0.01):
    """
    Rk4 method 
    dx1dt = lambda x1,x2,t : f1(x1,x2,t)
    dx2dt = lambda x1,x2,t : f2(x1,x2,t)
    return np array x1,x2,t
    """
    n=int((t_final-t_initial)/dt)
    if n<0:
        n=-n
        dt=-dt
    x1=np.zeros(n+1)
    x2=np.zeros(n+1)
    t=np.zeros(n+1)
    t[0]=t_initial
    x1[0]=x1_initial
    x2[0]=x2_initial
    for i in range(n):
        k1_x1,k1_x2=dx1dt(x1[i],x2[i],t[i]),dx2dt(x1[i],x2[i],t[i])
        k2_x1,k2_x2=dx1dt(x1[i]+dt/2*k1_x1,x2[i]+dt/2*k1_x2,t[i]+dt/2),dx2dt(x1[i]+dt/2*k1_x1,x2[i]+dt/2*k1_x2,t[i]+dt/2)
        k3_x1,k3_x2=dx1dt(x1[i]+dt/2*k2_x1,x2[i]+dt/2*k2_x2,t[i]+dt/2),dx2dt(x1[i]+dt/2*k2_x1,x2[i]+dt/2*k2_x2,t[i]+dt/2)
        k4_x1,k4_x2=dx1dt(x1[i]+dt*k3_x1,x2[i]+dt*k3_x2,t[i]+dt),dx2dt(x1[i]+dt*k3_x1,x2[i]+dt*k3_x2,t[i]+dt)
        x1[i+1]=x1[i]+dt*(k1_x1/6+k2_x1/3+k3_x1/3+k4_x1/6)
        x2[i+1]=x2[i]+dt*(k1_x2/6+k2_x2/3+k3_x2/3+k4_x2/6)
        t[i+1]=t[i]+dt
    return x1,x2,t


def dx1_dt_dx2_dt_rk2(t_initial,t_final,x1_initial,x2_initial,dx1dt,dx2dt,dt=0.01):
    """
    Rk4 method 
    dx1dt = lambda x1,x2,t : f1(x1,x2,t)
    dx2dt = lambda x1,x2,t : f2(x1,x2,t)
    return np array x1,x2,t
    """
    n=int((t_final-t_initial)/dt)
    if n<0:
        n=-n
        dt=-dt
    x1=np.zeros(n+1)
    x2=np.zeros(n+1)
    t=np.zeros(n+1)
    t[0]=t_initial
    x1[0]=x1_initial
    x2[0]=x2_initial
    for i in range(n):
        k1_x1,k1_x2=dx1dt(x1[i],x2[i],t[i]),dx2dt(x1[i],x2[i],t[i])
        k2_x1,k2_x2=dx1dt(x1[i]+dt/2*k1_x1,x2[i]+dt/2*k1_x2,t[i]+dt/2),dx2dt(x1[i]+dt/2*k1_x1,x2[i]+dt/2*k1_x2,t[i]+dt/2)
        x1[i+1]=x1[i]+dt*k2_x1
        x2[i+1]=x2[i]+dt*k2_x2
        t[i+1]=t[i]+dt
    return x1,x2,t


def d2x_dt2_rk4(t_initial,t_final,x_initial,xd_initial,dx2dt2,dt=0.01):
    """
    xd: dx/dt
    dx2dt2= lambda x,xd,t :f1(x,xd,t)
    return np array x,xd,t
    """
    x,xd,t=dx1_dt_dx2_dt_rk4(t_initial,t_final,x_initial,xd_initial,dx1dt=lambda x1,x2,t:x2,dx2dt=dx2dt2,dt=dt)
    return x,xd,t

def d2x_dt2_rk2(t_initial,t_final,x_initial,xd_initial,dx2dt2,dt=0.01):
    """
    xd: dx/dt
    dx2dt2= lambda x,xd,t :f1(x,xd,t)
    return np array x,xd,t
    """
    x,xd,t=dx1_dt_dx2_dt_rk2(t_initial,t_final,x_initial,xd_initial,dx1dt=lambda x1,x2,t:x2,dx2dt=dx2dt2,dt=dt)
    return x,xd,t

def d2x_dt2_boundary(t_initial,t_boundary,t_final,x_initial,x_boundary,d2xdt2,guess1=1,guess2=2,dt=0.01,method='rk4'):
    """
    xd: dx/dt
    dx2dt2= lambda x,xd,t :f1(x,xd,t)
    return np array x,xd,t
    """ 
    if method=='rk4':
        x1,xd1,t1=d2x_dt2_rk4(t_initial,t_boundary,x_initial,guess1,d2xdt2,dt)
    elif method=='rk2':
        x1,xd1,t1=d2x_dt2_rk2(t_initial,t_boundary,x_initial,guess1,d2xdt2,dt)
    x_boundary1=x1[-1]
    if x_boundary1==x_boundary:
        if method=='rk4':
            return d2x_dt2_rk4(t_initial,t_final,x_initial,guess1,d2xdt2,dt)
        elif method=='rk2':
            return d2x_dt2_rk2(t_initial,t_final,x_initial,guess1,d2xdt2,dt)
    if method=='rk4':
        x2,xd2,t2=d2x_dt2_rk4(t_initial,t_boundary,x_initial,guess2,d2xdt2,dt)
    elif method=='rk2':
        x2,xd2,t2=d2x_dt2_rk2(t_initial,t_boundary,x_initial,guess2,d2xdt2,dt)
    x_boundary2=x2[-1]
    if x_boundary2==x_boundary:
        if method=='rk4':
            return d2x_dt2_rk4(t_initial,t_final,x_initial,guess2,d2xdt2,dt)
        elif method=='rk2':
            return d2x_dt2_rk2(t_initial,t_final,x_initial,guess2,d2xdt2,dt)
    desired_guess=guess1+(guess2-guess1)*(x_boundary-x_boundary1)/(x_boundary2-x_boundary1)
    if method=='rk4':
        return d2x_dt2_rk4(t_initial,t_final,x_initial,desired_guess,d2xdt2,dt)
    elif method=='rk2':
        return d2x_dt2_rk2(t_initial,t_final,x_initial,desired_guess,d2xdt2,dt)
