#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import h5py

# ## Validation Strategies

#Objective Functions to minimize with Bayesian Optimization

def SSV(x):
    # Single Shot Validation
    
    global rho, sigma_in, tikh_opt, k, ti
    rho      = 10**x[0]
    sigma_in = 10**x[1]
    
    lenn     = len(tikh)
    Mean     = np.zeros(lenn)
    
    print('time:', -ti + time.time()) #check for computational cost
    ti       = time.time()
        
    #Train
    Xa_train, Wout = train(U_washout, U_t, Y_t, tikh, sigma_in, rho)[:2]

    #evaluate the networks optimized with different tikhonov parameters
    for j in range(lenn):
            #Validate
            Yh_val  = closed_loop(N_val, Xa_train[0,-1], Wout[j], sigma_in, rho)[0][1:]
            kh_val  = 0.5*np.linalg.norm(Yh_val, axis=1)**2
            Mean[j] = np.log10(np.mean((kh_val-k_v)**2)) #minimize with respect to kinetic energy


    a           = np.argmin(Mean) #select the optimal tikhonov parameter
    tikh_opt[k] = tikh[a]
    k          +=1
    print(k,Mean, Mean[a])
        
    return Mean[a]


def RVC(x):
    #Recycle Validation
    
    global rho, sigma_in, tikh_opt, k, ti
    rho      = 10**x[0]
    sigma_in = 10**x[1]
        
    print('time:', -ti + time.time()) #check for computational cost
    ti       = time.time()
        
    lenn     = tikh.size
    Mean     = np.zeros(lenn)
    
    #Train using tv: training+val
    Xa_train, Wout, LHS0, RHS0 = train(U_washout, U_tv, Y_tv, tikh, sigma_in, rho)

    # N_tval is the number of time series used for validation
    for kk in range(N_tval):

        #Different validation intervals in each time series
        for i in range(N_fo):

            p      = N_in + i*N_fw
            k_val  = 0.5*np.linalg.norm(UU[kk,N_washout + p : N_washout + p + N_val], axis=1)**2 #kinetic energy of the true data
            
            # differrent tikhnov parameters (no need to retrain Wout per each fold)
            for j in range(lenn):
                #Validate
                Yh_val    = closed_loop(N_val-1, Xa_train[kk,p], Wout[j], sigma_in, rho)[0]
                kh_val    = 0.5*np.linalg.norm(Yh_val, axis=1)**2
        
                Mean[j]  += np.log10(np.mean((kh_val-k_val)**2))
                
    a           = np.argmin(Mean) #select the optimal tikhonov parameter
    tikh_opt[k] = tikh[a]
    k          +=1
    print(k,Mean/N_fo/N_tval, Mean[a]/N_fo/N_tval)

    return Mean[a]/N_fo/N_tval #average over intervals and time series