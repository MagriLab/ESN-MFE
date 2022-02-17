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


def closed_loop_stats(N, x0, Wout, sigma_in, rho):
    """ Advances ESN in closed-loop, stops it if the flow has laminarized (for computational efficiency)
        Args:
            N: number of time steps
            x0: initial reservoir state
            Wout: output matrix
        Returns:
            time series of prediction
            final augmented reservoir state
    """
    xa = x0.copy()
    Yh = np.zeros((N+1, dim))
    Yh[0] = np.dot(xa, Wout)
    for i in np.arange(1,N+1):
        xa = step(xa[:N_units], Yh[i-1], sigma_in, rho)
        Yh[i] = np.dot(xa, Wout)
        if Yh[i,0] >= 1:
            break

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
    
    N   = U_train[0].shape[0]    
    Xa  = np.zeros((U_washout.shape[0], N+1, N_units+1))
    
    for i in range(U_washout.shape[0]):
        
        ## washout phase
        xf_washout = open_loop(U_washout[i], np.zeros(N_units), sigma_in, rho)[-1,:N_units]

        ## open-loop train phase
        Xa[i] = open_loop(U_train[i], xf_washout, sigma_in, rho)
    
        ## Ridge Regression
        LHS  += np.dot(Xa[i,1:].T, Xa[i,1:])
        RHS  += np.dot(Xa[i,1:].T, Y_train[i])
    
    Wout = np.linalg.solve(LHS + tikh*np.eye(N_units+1), RHS)

    return Xa[0], Wout, LHS, RHS


# In[2]:


# def predictability_horizon(xa, Y, Wout):
#     """ Compute predictability horizon. It evolves the network until the
#         error is greater than the threshold. Before that it initialises
#         the network by running a washout phase.
        
#         Args:
#             threshold: error threshold
#             U_washout: time series for washout
#             Y: time series to compare prediction
        
#         Returns:
#             predictability horizon (in time units, not Lyapunov times)
#             time series of normalised error
#             time series of prediction
#     """
    
#     # calculate denominator of the normalised error
#     kin   = 0.5*np.linalg.norm(Y, axis=1)**2
# #     err_z = np.sqrt(np.mean(kin**2))
#     err_d = np.mean(np.sum((Y[:,:dim])**2, axis=1))
    
#     N     = Y.shape[0]
#     E     = np.zeros(N+1)
#     E_z   = np.zeros(N+1)
#     Yh    = np.zeros((N, len(idx1)))
#     kinh  = np.zeros(N)
    
#     Yh[0]   = np.dot(xa, Wout)
#     kinh[0] =0.5*np.linalg.norm(Yh[0])**2
    
#     pr_idx = np.zeros(N)
#     pr_idd = np.zeros(N)
    
#     pr_idx[-1] = 1
#     pr_idd[-1] = 1
    
#     for i in range(1, N):
#         # advance one step
#         xa       = step(xa[:N_units], Yh[i-1][:dim])
#         Yh[i]    = np.dot(xa, Wout)
#         kinh[i]  = 0.5*np.linalg.norm(Yh[i])**2
    
#         # calculate error
#         E_z[i]  = np.abs(kinh[i]-kin[i])
#         err_n   = np.sum(((Yh[i,:dim]-Y[i,:dim]))**2)
#         E[i]    = np.sqrt(err_n/err_d)

#         if E_z[i] > threshold:
#             break
        
#         if E_z[i] > tt:
#             pr_idx[i] = 1
            
#         if E[i]   > 0.2:
#             pr_idd[i] = 1
    
#     a = np.nonzero(pr_idx)[0][0]/N_lyap
#     b = np.nonzero(pr_idd)[0][0]/N_lyap

#     t = np.arange(i)/N_lyap
# #     fig, ax1 = plt.subplots(1,2)
    
# #     ax=plt.subplot(1,2,1)
# #     plt.annotate('{:.2f}'.format(a), xy=(0, 1), xytext=(5, -5), va='top', ha='left',
# #              xycoords='axes fraction', textcoords='offset points')
# #     plt.ylim(1e-4,threshold)
# #     plt.axhline(.2)
# #     ax.set_ylabel('$E$')
# #     plt.plot(t,E_z[:i], label='$E_k$')
# #     plt.plot(t,E[:i], label='$E$')
# #     plt.legend()
    
# #     ax = plt.subplot(1,2,2)
# #     ax.set_ylabel('$k$')
#     if is_plot:
#         plt.axhline(ee,linewidth=3, alpha=0.3)
#         plt.axvline(a,linewidth=3, alpha=0.3)
#         plt.plot(t,kin[:i], label='True')
#         plt.plot(t,kinh[:i], label='ESN')
#         plt.legend(fontsize=20)
#         plt.show()
            
#     return a, b

def predictability_horizon_k(xa, Y, Wout, sigma_in, rho):
    """ Compute predictability horizon for the kinetic energy. It evolves the network until the
        error is greater than the threshold. Before that it initialises
        the network by running a washout phase.
    """
    
    # calculate kinetic energy of the data
    kin   = 0.5*np.linalg.norm(Y, axis=1)**2
    
    #initialize
    N     = Y.shape[0]
    E     = np.zeros(N+1)
    Yh    = np.zeros((N, Ndim))
    kinh  = np.zeros(N)
    
    Yh[0]   = np.dot(xa, Wout)
    kinh[0] = 0.5*np.linalg.norm(Yh[0])**2
    
    for i in range(1, N):
        # advance one step the ESN
        xa       = step(xa[:N_units], Yh[i-1][:dim], sigma_in, rho)
        Yh[i]    = np.dot(xa, Wout)
        kinh[i]  = 0.5*np.linalg.norm(Yh[i])**2
    
        # calculate error
        E[i]  = np.abs(kinh[i]-kin[i])
        
        #stop if error is larger than threshold    
        if E[i] > threshold:
            break
            
    return i/N_lyap
