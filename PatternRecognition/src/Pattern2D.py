from Logger import *
from utils import *

import matplotlib.patches as ptch
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import random
import json
import pywt

class Pattern2D:
    
    def __init__(self, pattern):
        pattern = np.array(pattern)
        self.M = pattern
        self.N1 = pattern.shape[0]+pattern.shape[0]%2-2
        self.N2 = pattern.shape[1]+pattern.shape[1]%2-2
        self.N = max(self.N1, self.N2)
        self.system = np.zeros((self.N, self.N))
        self.right = np.zeros((self.N, 1))
        self.jacobian = np.zeros((self.N, self.N))
        self.buildSystem()
        
        if not self.checkForSaveFilters():
            self.buildFilter()
            self.saveFilterData()
        # self.buildFilter()
    
    def noise(self, details, eps=1e-3):
        for i in range(details.shape[0]):
            for j in range(details.shape[1]):
                details[i][j] = 0 if abs(details[i][j]) < eps else details[i][j]
        return details
    
    
    def buildSystem(self):
        n = self.N
        
        self.system[n//2][0] = self.jacobian[n//2][0] = 1
        for i in range(n//2, n-4):
            for j in range(n):
                self.jacobian[i][j] = self.system[i][j] = (j+1)**(i-n//2)
        
        self.right[0][0] = 1
        
        
    def checkForSaveFilters(self):
        try:
            logger.info("Checking for filters saved")
            with open('filters_data.json', 'r') as json_file:
                data = json.load(json_file)
            
            for i in range(len(data.keys())//2):
                M = np.array(data[f'patron{formatToString(i, 3)}'], np.longdouble)
                if self.M.shape[0] == len(M) and self.M.shape[1] == len(M[0]):
                    F = self.M - M
                    norm = sum([sum([v*v for v in row]) for row in F])
                    if norm < 1e-6:
                        logger.info("Loading filter bank")
                        self.wavelet = pywt.Wavelet(filter_bank = data[f'filter_bank{formatToString(i, 3)}'])
                        return True
            return False
        except:
            logger.warning("No filters saved for the pattern")
            return False
    
    
    def saveFilterData(self):
        logger.info("Saving new filter bank")
        try:
            with open('filters_data.json', 'r') as json_file:
                data = json.load(json_file)
            
            n = len(data.keys())//2
            data[f'patron{formatToString(n, 3)}'] = [[float(v) for v in row] for row in self.M]
            data[f'filter_bank{formatToString(n, 3)}'] = self.wavelet.filter_bank
        except:
            data = {
                'patron000': [[float(v) for v in row] for row in self.M],
                'filter_bank000': self.wavelet.filter_bank
            }
        with open('filters_data.json', 'w') as json_file:
            json.dump(data, json_file)
    
    
    def func(self, x):
        n, m = self.N, self.M
        N1, N2 = self.N1, self.N2
        x = np.array(x).reshape((n, 1))
        
        # Orthogonality conditions
        for i in range(n//2):
            for j in range(n-2*i):
                self.system[i][j] = x[i*2+j][0]
        
        # Matching conditions
        for i1 in range(2):
            for i2 in range(2):
                suma = np.zeros((n,))
                for j in range(1,N2+1):
                    suma[j%n] += sum([x[i%n]*m[i-i2][j-i1] for i in range(1,N1+1)])
                self.system[n-4+i1*2+i2] = suma
                
        return self.system.dot(x)-self.right

    
    def jac(self, x):
        n, m = self.N, self.M
        N1, N2 = self.N1, self.N2
        x = np.array(x).reshape((n, 1))
        
        # Orthogonality conditions
        for i in range(n//2):
            for j in range(n-2*i):
                self.jacobian[i][j] = x[i*2+j][0] * (2 if i == 0 else 1)
        
        # Matching conditions
        for i1 in range(2):
            for i2 in range(2):
                suma = np.zeros((n,))
                for j in range(1,N2+1):
                    suma[j%n] += sum([x[i%n]*m[i-i2][j-i1]*(2 if (i-j)%n==0 else 1) for i in range(1,N1+1)])
                self.jacobian[n-4+i1*2+i2] = suma
        
        return self.jacobian
    
    
    def buildFilter(self):
        n = self.N
        x0 = [random.random() for _ in range(n)]
        q = gekko_2D(self.M, homotopia(self.func, self.jac, x0))
        q = q/np.linalg.norm(q)
        logger.info(f'Checking solution: \n{self.func(q)}')
        p = np.array([(-1)**k * q[(n-k+1)%n] for k in range(n)])
        q_ = np.array([q[(n-k)%n] for k in range(n)])
        p_ = np.array([p[(n-k)%n] for k in range(n)])
        
        logger.info("Wavelet base created")
        self.wavelet = pywt.Wavelet(filter_bank = [p_, q_, p, q])
        
        
    def load_detect(self, F = None, path = None, plot=False):
        if F is None:
            F = np.array(pydicom.dcmread(path, force=True).pixel_array) / (1<<16)
        logger.info("Image loaded")
        
        N1, N2 = self.N1, self.N2
        D1, D2 = F.shape
        p_, q_ = self.wavelet.dec_lo, self.wavelet.dec_hi
        m = self.M
        
        # full matching
        logger.info("Applying DWT for two dimensions...")
        # res = pywt.dwt2(F, self.wavelet, mode="zpd")
        cA, cH, cV, cD, _valid = [np.zeros(((D1+1)//2, (D2+1)//2)) for _ in range(5)]
        for i in range(cA.shape[0]):
            for j in range(cA.shape[1]):
                for k1 in range(N1):
                    for k2 in range(N2):
                        if F[(2*i+D1-k1)%D1][(2*j+D2-k2)%D2] > 0:
                            _valid[i][j] += 1
                        cD[i][j] += q_[k1] * q_[k2] * F[(2*i+D1-k1)%D1][(2*j+D2-k2)%D2] * (1<<16)
                        cH[i][j] += q_[k1] * p_[k2] * F[(2*i+D1-k1)%D1][(2*j+D2-k2)%D2]
                        cV[i][j] += p_[k1] * q_[k2] * F[(2*i+D1-k1)%D1][(2*j+D2-k2)%D2]
                        cA[i][j] += p_[k1] * p_[k2] * F[(2*i+D1-k1)%D1][(2*j+D2-k2)%D2]
        
        # only patron matching

        if plot:
            logger.info("Showing results...")
            fig, ax = plt.subplots(3, 2)
            ax[0, 0].imshow(cA, cmap=plt.cm.bone)
            ax[0, 1].imshow(cH, cmap=plt.cm.bone)
            ax[1, 0].imshow(cV, cmap=plt.cm.bone)
            ax[1, 1].imshow(cD, cmap=plt.cm.bone)
            
            sim, alpha = [], 0.1
            for r in range(cD.shape[0]):
                for c in range(cD.shape[1]):
                    val = np.e**(-abs(cD[r][c])**alpha)
                    if _valid[r][c] < self.M.shape[0]*self.M.shape[1]/4:
                        val = 0
                    sim.append(val)
            
            patches, umbral = [], 0.4
            maximum, patch = 0, None
            for i in range(cD.shape[0]):
                for j in range(cD.shape[1]):
                    # if sim[cD.shape[1]*i+j] > maximum and sim[cD.shape[1]*i+j] < 0.96:
                    #     maximum = max(maximum, sim[cD.shape[1]*i+j])
                    #     patch = ptch.Rectangle( ((j*2-m.shape[1]+1+F.shape[1])%F.shape[1],
                    #                             (i*2-m.shape[0]+1+F.shape[0])%F.shape[0]),
                    #                         m.shape[1]-1,
                    #                         m.shape[0]-1,
                    #                         linewidth=1,
                    #                         edgecolor='red',
                    #                         facecolor="none")
                    if sim[cD.shape[1]*i+j] > umbral and sim[cD.shape[1]*i+j] < 0.96:
                        rect = ptch.Rectangle( ((j*2-m.shape[1]+1+F.shape[1])%F.shape[1],
                                                (i*2-m.shape[0]+1+F.shape[0])%F.shape[0]),
                                            m.shape[1]-1,
                                            m.shape[0]-1,
                                            linewidth=1,
                                            edgecolor='red',
                                            facecolor="none")
                        patches.append(rect)
            
            ax[2, 0].imshow(F, cmap=plt.cm.bone)            
            # ax[2, 0].add_patch(patch)
            for patch in patches:
                ax[2, 0].add_patch(patch)
            
            ax[2, 1].plot(np.linspace(0, cD.size-1, cD.size), sim, color = "blue")
            ax[2, 1].grid()
        
        return (cA, (cH, cV, cD))