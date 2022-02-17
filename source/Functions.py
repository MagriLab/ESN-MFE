import numpy as np
import matplotlib.pyplot as plt
import h5py

## ESN with bias architecture

def step(x_pre, u, sigma_in, rho):
    """ Advances one ESN time step.
        Args:
            x_pre: reservoir state
            u: input
        Returns:
            new augmented state (new state with bias_out appended)
    """
    # input is normalized and input bias added
    u_augmented = np.hstack((u/norm, np.array([bias_in]))) 
    # hyperparameters are explicit here
    x_post      = np.tanh(np.dot(u_augmented*sigma_in, Win) + rho*np.dot(x_pre, W)) 
    # output bias added
    x_augmented = np.hstack((x_post, np.array([bias_out])))

    return x_augmented

def open_loop(U, x0, sigma_in, rho):
    """ Advances ESN in open-loop.
        Args:
            U: input time series
            x0: initial reservoir state
        Returns:
            time series of augmented reservoir states
    """
    N  = U.shape[0]
    Xa = np.empty((N+1, N_units+1))
    Xa[0] = np.hstack((x0, np.array([bias_out])))
    for i in np.arange(1,N+1):
        Xa[i] = step(Xa[i-1,:N_units], U[i-1], sigma_in, rho)

    return Xa

def closed_loop(N, x0, Wout, sigma_in, rho):
    """ Advances ESN in closed-loop.
        Args:
            N: number of time steps
            x0: initial reservoir state
            Wout: output matrix
        Returns:
            time series of prediction
            final augmented reservoir state
    """
    xa = x0.copy()
    Yh = np.empty((N+1, dim))
    Yh[0] = np.dot(xa, Wout)
    for i in np.arange(1,N+1):
        xa = step(xa[:N_units], Yh[i-1], sigma_in, rho)
        Yh[i] = np.dot(xa, Wout)

    return Yh, xa

def train(U_washout, U_train, Y_train, tikh, sigma_in, rho):
    """ Trains ESN.
        Args:
            U_washout: washout input time series
            U_train: training input time series
            tikh: Tikhonov factor
        Returns:
            time series of augmented reservoir states
            optimal output matrix
    """
    
    LHS = 0
    RHS = 0
    
    N  = U_train[0].shape[0]    
    Xa  = np.zeros((U_washout.shape[0], N+1, N_units+1))
    
    for i in range(U_washout.shape[0]):
        
        ## washout phase
        xf_washout = open_loop(U_washout[i], np.zeros(N_units), sigma_in, rho)[-1,:N_units]

        ## open-loop train phase
        Xa[i] = open_loop(U_train[i], xf_washout, sigma_in, rho)
    
        ## Ridge Regression
        LHS  += np.dot(Xa[i,1:].T, Xa[i,1:])
        RHS  += np.dot(Xa[i,1:].T, Y_train[i])
    
    #solve linear system for each Tikhonov parameter
    Wout = np.zeros((tikh.size, N_units+1,dim))
    for j in np.arange(tikh.size):
        Wout[j] = np.linalg.solve(LHS + tikh[j]*np.eye(N_units+1), RHS)

    return Xa[:N_tval], Wout, LHS, RHS