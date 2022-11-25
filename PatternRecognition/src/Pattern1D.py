from asyncio.log import logger
from utils import *

import matplotlib.pyplot as plt
from gekko import GEKKO
import numpy as np
import random
import pywt

class Pattern1D:
    
    def __init__(self, pattern : np.ndarray):
        self.M = pattern
        self.N = len(pattern) - 1
        self.system = np.zeros((self.N, self.N))
        self.right = np.zeros((self.N, 1))
        self.jacobian = np.zeros((self.N, self.N))
        self.buildSystem()
        self.buildFilter()
    
    
    def buildSystem(self):
        n, m = self.N, self.M
        
        self.system[n//2][0] = self.jacobian[n//2][0] = 1
        for i in range(n//2, n-2):
            for j in range(n):
                self.jacobian[i][j] = self.system[i][j] = (j+1)**(i-n//2)
        
        for i in range(n-2, n):
            for j in range(1, n+1):
                self.jacobian[i][j%n] = self.system[i][j%n] = m[j+n-2-i]
        
        self.right[0][0] = 1
    
    
    def norm2(self, x):
        return sum([f[0]**2 for f in self.func(x)])
    
    
    def grad(self, x):
        return 2*self.jac(x).T.dot(self.func(x))
            
        
    def func(self, x):
        """
        Evaluate f(x)

        Args:
            x (C^n): Domain

        Returns:
            f (C^n): Result of f(x)
        """
        n = self.N
        x = np.array(x).reshape((n, 1))
        
        for i in range(n//2):
            for j in range(n-2*i):
                self.system[i][j] = x[i*2+j][0]
        
        return self.system.dot(x)-self.right
        
    
    def jac(self, x):
        n = self.N
        x = np.array(x).reshape((n, 1))
        
        for i in range(n//2):
            for j in range(n-2*i):
                self.jacobian[i][j] = x[i*2+j][0] * (2 if i == 0 else 1)
        
        return self.jacobian
        
    
    def buildFilter(self):
        n = self.N
        x0 = [random.random() for _ in range(n)]
        q = gekko_1D(self.M, homotopia(self.func, self.jac, x0))
        logger.info(f'Checking solution: \n{q}\n{self.func(q)}')
        q = q * 2/np.linalg.norm(q)
        p = np.array([(-1)**k * q[(n-k+1)%n] for k in range(n)])
        q_ = np.array([q[(n-k)%n] for k in range(n)])
        p_ = np.array([p[(n-k)%n] for k in range(n)])
        
        self.p_, self.q_ = p_, q_
        
        self.wavelet = pywt.Wavelet(filter_bank = [p_, q_, p, q])
    
    
    def detectPatron(self, sample, plot=False):
        (cA, cD) = pywt.dwt(sample, self.wavelet, mode = "zpd")
        
        if plot:
            N = len(sample)
            sim, alpha = [], 0.1
            for i in list(cD):
                sim.append(np.e**(-abs(i)**alpha))
            
            X1 = np.linspace(0, N-1, N)
            X2 = np.linspace(0, cD.shape[0]-1, cD.shape[0])
            _, ax = plt.subplots(1, 2)
            ax[0].plot(X1, sample, color="red")
            ax[0].grid()
            ax[1].plot(X2, list(cD), color="orange")
            ax[1].scatter(X2, list(cD), color="orange")
            ax[1].plot(X2, sim, color="blue")
            ax[1].grid()
            
        return cA, cD