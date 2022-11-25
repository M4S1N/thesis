from sympy import maximum
from Logger import *

from gekko import GEKKO
import numpy as np
import random

def formatToString(x : int, n : int):
    string = str(x)
    while len(string) < n:
        string = "0" + string
    return string

def printlist(a):
    print("[")
    for row in a:
        s = "["
        for v in row:
            s = f"{s} {v}"
        s += "]"
        print(s)
    print("]")
    
def normalize(nplist):
    def get_maximum(nplist : np.ndarray):
        if len(nplist.shape) == 1:
            maximum = -1e10
            for val in nplist:
                if abs(val) > maximum:
                    maximum = abs(val)
            return maximum
        elif len(nplist) == 1:
            return get_maximum(nplist[0])
        return max(get_maximum(nplist[0]), get_maximum(nplist[1:]))
    return nplist / get_maximum(nplist)

def newton(F, J, x, eps = 1e-8):
    n = len(x)
    x = np.array(x).reshape((n, 1))
    for _ in range(1000):
        y = -np.linalg.inv(J(x)).dot(F(x))
        x = x + y
        if np.linalg.norm(y) < eps:
            logger.info(f"Solution finded with newton")
            return x.reshape((n,))
    logger.info(f"Not finded")
    return x.reshape((n,))

def cuasi_newton(F, J, x, eps = 1e-8):
    n = len(x)
    x = np.array(x).reshape((n, 1))
    v = F(x)
    A = np.linalg.inv(J(x))
    s = -A.dot(v)
    x = x + s
    for _ in range(1000):
        w = v
        v = F(x)
        y = v - w
        z = -A.dot(y)
        p = -s.T.dot(z)[0][0]
        ut = s.T.dot(A)
        A = A + 1/p*(s+z).dot(ut)
        s = -A.dot(v)
        x = x + s
        if np.linalg.norm(s) < eps:
            logger.info(f"Solution finded with cuasi newton")
            return x.reshape((n,))
    logger.info(f"Not finded")
    return x.reshape((n,))

def gradient_descen(F, G, x, eps = 1e-8):
    n = len(x)
    x = np.array(x).reshape((n, 1))
    for i in range(1000):
        z = G(x)
        z0 = np.linalg.norm(z)
        if z0 < eps:
            logger.info(f"Solution finded with gradient descens |dG| < eps. func = {F(x)}")
            return x.reshape((n,))
        z = z / z0
        a1, a3 = 0, 1
        f1 = F(x - a1*z)
        f3 = F(x - a3*z)
        while f3 >= f1:
            a3 /= 2
            f3 = F(x - a3*z)
            if a3 < eps / 2:
                logger.info(f"Solution finded with gradient descens a3 < eps/2. func = {F(x)}")
                return x.reshape((n,))
        a2 = a3 / 2
        f2 = F(x - a2*z)
        h1 = (f2 - f1) / a2
        h2 = (f3 - f2) / (a3 - a2)
        h3 = (h2 - h1) / a3
        a0 = 0.5*(a2 - h1/h3)
        f0 = F(x - a0*z)
        _,a = min((f0,a0), (f3,a3))
        x = x - a*z
        if abs(F(x)-f1) < eps:
            logger.info(f"Solution finded with gradient descens |F-F1| < eps. func = {F(x)}")
            return x.reshape((n,))
        if np.linalg.norm(a*z) < eps:
            logger.info(f"Solution finded with gradient descens |xk - xk-1| < eps. func = {F(x)}")
            return x.reshape((n,))
    return x.reshape((n,))

def homotopia(F, J, x, eps=1e-8):
    n, N = len(x), 250
    x = np.array(x).reshape((n, 1))
    h = 1/N
    b = -h * F(x)
    
    for i in range(N):
        A = J(x)
        k1 = np.linalg.inv(A).dot(b)
        A = J(x + 0.5*k1)
        k2 = np.linalg.inv(A).dot(b)
        A = J(x + 0.5*k2)
        k3 = np.linalg.inv(A).dot(b)
        A = J(x + k3)
        k4 = np.linalg.inv(A).dot(b)
        x = x + (k1 + 2*k2 + 2*k3 + k4) / 6
    return x.reshape((n,))

def gekko_1D(pattern, x0):
    n = len(x0)
    m = pattern
    
    model = GEKKO(remote=False)
    a = [model.Var(xi) for xi in x0]
    
    # Orthogonality conditions
    for i in range(n//2):
        if i == 0:
            model.Equation(model.sum([a[k]**2 for k in range(n)]) == 1)
        else:
            model.Equation(model.sum([a[k]*a[k+2*i] for k in range(n-2*i)]) == 0)
    
    # Vanishing moments
    for i in range(n//2 - 2):
        if i == 0:
            model.Equation(model.sum([a[k] for k in range(n)]) == 0)
        else:
            model.Equation(model.sum([a[k]*(k+1)**i for k in range(n)]) == 0)
            
    # Matching conditions
    model.Equation(model.sum([a[(n-k)%n]*m[n-k] for k in range(n)]) == 0)
    model.Equation(model.sum([a[(n-k)%n]*m[n-k-1] for k in range(n)]) == 0)
    
    model.options.TIME_SHIFT = 0
    model.options.COLDSTART = 2
    model.solve(disp=False)
    return np.array([x.value[0] for x in a])

def gekko_2D(pattern, x0):
    n, m = len(x0), pattern
    model = GEKKO(remote=False)
    a = [model.Var(x0[i]) for i in range(n)]
    
    # Orthogonality conditions
    for i in range(n//2):
        if i == 0:
            eq = model.sum([a[k]**2 for k in range(n)])
            model.Equation(eq == 1)
        else:
            eq = model.sum([a[k]*a[k+2*i] for k in range(n-2*i)])
            model.Equation(eq == 0)
    
    # # Vanishing moments
    for i in range(n//2 - 4):
        if i == 0:
            eq = model.sum([a[k] for k in range(n)])
            model.Equation(eq == 0)
        else:
            eq = model.sum([a[k]*(k+1)**i for k in range(n)])
            model.Equation(eq == 0)
    
    # Matching conditions
    for i in range(4):
        eq = model.sum([model.sum([a[k1%n]*a[k2%n]*m[k1-i//2][k2-i%2] for k2 in range(1,n+1)]) for k1 in range(1,n+1)])
        model.Equation(eq == 0)
    
    logger.info(f"Finding filter of size {n}...")
    model.options.TIME_SHIFT = 0
    model.options.COLDSTART = 2
    model.solve(disp=False)
    
    return np.array([x.value[0] for x in a])
        

if __name__ == '__main__':
    def func(x):
        x1, x2, x3 = np.array(x).reshape((3,))
        f1 = 3*x1 - np.cos(x2*x3) - 1/2
        f2 = x1**2 - 81*(x2+0.1)**2 + np.sin(x3) + 1.06
        f3 = np.exp(-x1*x2) + 20*x3 + (10*np.pi-3)/3
        return np.array([f1, f2, f3]).reshape((3,1))
    
    def norm2(x):
        return sum([f**2 for f in func(x)])
    
    def jac(x):
        x1, x2, x3 = np.array(x).reshape((3,))
        j1 = [3, x3*np.sin(x2*x3), x2*np.sin(x2*x3)]
        j2 = [2*x1, -162*(x2+0.1), np.cos(x3)]
        j3 = [-x2*np.exp(-x1*x2), -x1*np.exp(-x1*x2), 20]
        return np.array([j1, j2, j3])
    
    def grad(x):
        return 2 * jac(x).T.dot(func(x))
    
    print(norm2(gradient_descen(norm2, grad, [0, 0, 0])))
        