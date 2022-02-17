import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import math
import h5py


def gsQR(M):
    ''' QR decomposition based on Gram-Schmidt '''
    Q = np.zeros(M.shape)
    R = np.zeros(M.shape)

    for j in range(M.shape[0]):
        v = M[:,j].copy()
        for i in range(j):
            R[i,j] = np.dot(Q[:,i], M[:,j])
            v -= R[i,j]*Q[:,i]
        R[j,j] = np.linalg.norm(v)
        Q[:,j] = v/R[j,j]

    return Q, R


def RK4(q0,dt,N,func):
    ''' 4th order RK for autonomous systems described by func '''

    global ii 

    q        = np.zeros((N+1,q0.shape[0]))
    q[0]     = q0
    k        = np.zeros(N+1)

    for i in 1+np.arange(N):

        k1   = dt * func(q[i-1])
        k2   = dt * func(q[i-1] + k1/2)
        k3   = dt * func(q[i-1] + k2/2)
        k4   = dt * func(q[i-1] + k3)

        q[i] = q[i-1] + (k1 + 2*k2 + 2*k3 + k4)/6

        if func == MFE:

            k[i] = 0.5*np.linalg.norm(q[i,:Ndim])**2

            #check laminarization
            if k[i] > 0.48 and i > N_tran:
                print('Laminarized', ii)
                ii += 1
                q[i,-1] = 100
                break


    return  q


def MFE(q):
    """
    Defines the differential equations for Moehlis_2004
    """
    a1, a2, a3, a4, a5, a6, a7, a8, a9 = q


    k1 = np.sqrt(alfa**2 + gamma**2)
    k2 = np.sqrt(gamma**2 + beta**2)
    k3 = np.sqrt(alfa**2 + beta**2 + gamma**2) 

    dqdt = np.array([beta**2/Re * (1. - a1) - np.sqrt(3/2)*beta*gamma/k3*a6*a8 + np.sqrt(3/2)*beta*gamma/k2*a2*a3,

         - ( 4/3*beta**2 + gamma**2) * a2/Re + 5/3*np.sqrt(2/3)*gamma**2/k1*a4*a6 - gamma**2/np.sqrt(6)/k1*a5*a7 -
         alfa*gamma*beta/np.sqrt(6)/k1/k3*a5*a8 - np.sqrt(3/2)*beta*gamma/k2 * (a1*a3 + a3*a9),

         - (beta**2 + gamma**2)/Re*a3 + 2/np.sqrt(6)*alfa*beta*gamma/k1/k2 * (a4*a7 + a5*a6) + 
         (beta**2*(3*alfa**2+gamma**2) - 3*gamma**2*(alfa**2+gamma**2))/np.sqrt(6)/k1/k2/k3*a4*a8,

         - (3*alfa**2 + 4*beta**2)/3/Re*a4 - alfa/np.sqrt(6)*a1*a5 - 10/3/np.sqrt(6)*alfa**2/k1*a2*a6 -
         np.sqrt(3/2)*alfa*beta*gamma/k1/k2*a3*a7 - np.sqrt(3/2)*alfa**2*beta**2/k1/k2/k3*a3*a8 - alfa/np.sqrt(6)*a5*a9,

         - (alfa**2 + beta**2)/Re*a5 + alfa/np.sqrt(6)*a1*a4 + alfa**2/np.sqrt(6)/k1*a2*a7 - 
         alfa*beta*gamma/np.sqrt(6)/k1/k3*a2*a8 + alfa/np.sqrt(6)*a4*a9 + 2/np.sqrt(6)*alfa*beta*gamma/k1/k2*a3*a6,

         - (3*alfa**2 + 4*beta**2 + 3*gamma**2)/3/Re*a6 + alfa/np.sqrt(6)*a1*a7 + np.sqrt(3/2)*beta*gamma/k3*a1*a8 +
         10/3/np.sqrt(6)*(alfa**2 - gamma**2)/k1*a2*a4 - 2*np.sqrt(2/3)*alfa*beta*gamma/k1/k2*a3*a5 + alfa/np.sqrt(6)*a7*a9 + np.sqrt(3/2)*beta*gamma/k3*a8*a9,

         - k3**2/Re*a7 - alfa/np.sqrt(6) * (a1*a6 + a6*a9) + (gamma**2 - alfa**2)/np.sqrt(6)/k1*a2*a5 + alfa*beta*gamma/np.sqrt(6)/k1/k2*a3*a4,

         - k3**2/Re*a8 + 2/np.sqrt(6)*alfa*beta*gamma/k3/k1*a2*a5 + gamma**2*(3*alfa**2 - beta**2 + 3*gamma**2)/np.sqrt(6)/k1/k2/k3*a3*a4,

         - 9*beta**2/Re*a9 + np.sqrt(3/2)*beta*gamma/k2*a2*a3 - np.sqrt(3/2)*beta*gamma/k3*a6*a8
         ])
    
    return dqdt


def MFE_Jac(q):
    """
    Defines the jacobian for the differential equations for Moehlis_2004

    Arguments:
        w :  vector of the state variables a_i
        p :  vector of the parameters:
    """
    a1, a2, a3, a4, a5, a6, a7, a8, a9 = q
    
    k1 = np.sqrt(alfa**2 + gamma**2)
    k2 = np.sqrt(gamma**2 + beta**2)
    k3 = np.sqrt(alfa**2 + beta**2 + gamma**2) 

    J = np.array([[-beta**2/Re, np.sqrt(3/2)*beta*gamma/k2*a3, np.sqrt(3/2)*beta*gamma/k2*a2, 0., 0., - np.sqrt(3/2)*beta*gamma/k3*a8, 0., - np.sqrt(3/2)*beta*gamma/k3*a6, 0.],

               [- np.sqrt(3/2)*beta*gamma/k2*a3, - (4/3*beta**2 + gamma**2)/Re, - np.sqrt(3/2)*beta*gamma/k2*(a1 +a9), 5/3*np.sqrt(2/3)*gamma**2/k1*a6, 
                - gamma**2/np.sqrt(6)/k1*a7 - alfa*gamma*beta/np.sqrt(6)/k1/k3*a8, 5/3*np.sqrt(2/3)*gamma**2/k1*a4, - gamma**2/np.sqrt(6)/k1*a5, - alfa*gamma*beta/np.sqrt(6)/k1/k3*a5, - np.sqrt(3/2)*beta*gamma/k2*a3],

               [0., 0., - (beta**2 + gamma**2)/Re, + 2/np.sqrt(6)*alfa*beta*gamma/k1/k2*a7 + (beta**2*(3*alfa**2+gamma**2) - 3*gamma**2*(alfa**2+gamma**2))/np.sqrt(6)/k1/k2/k3*a8, 2/np.sqrt(6)*alfa*beta*gamma/k1/k2*a6, 
                2/np.sqrt(6)*alfa*beta*gamma/k1/k2*a5, 2/np.sqrt(6)*alfa*beta*gamma/k1/k2*a4, (beta**2*(3*alfa**2+gamma**2) - 3*gamma**2*(alfa**2+gamma**2))/np.sqrt(6)/k1/k2/k3*a4, 0.],
               
               [- alfa/np.sqrt(6)*a5, - 10/3/np.sqrt(6)*alfa**2/k1*a6, - np.sqrt(3/2)*alfa*beta*gamma/k1/k2*a7 - np.sqrt(3/2)*alfa**2*beta**2/k1/k2/k3*a8, - (3*alfa**2 + 4*beta**2)/3/Re,
                - alfa/np.sqrt(6)*a1 - alfa/np.sqrt(6)*a9, - 10/3/np.sqrt(6)*alfa**2/k1*a2, - np.sqrt(3/2)*alfa*beta*gamma/k1/k2*a3, - np.sqrt(3/2)*alfa**2*beta**2/k1/k2/k3*a3, - alfa/np.sqrt(6)*a5],

               [alfa/np.sqrt(6)*a4, + alfa**2/np.sqrt(6)/k1*a7 - alfa*beta*gamma/np.sqrt(6)/k1/k3*a8, 2/np.sqrt(6)*alfa*beta*gamma/k1/k2*a6, alfa/np.sqrt(6)*(a1 + a9), - (alfa**2 + beta**2)/Re,
                2/np.sqrt(6)*alfa*beta*gamma/k1/k2*a3, alfa**2/np.sqrt(6)/k1*a2, - alfa*beta*gamma/np.sqrt(6)/k1/k3*a2, alfa/np.sqrt(6)*a4],

               [alfa/np.sqrt(6)*a7 + np.sqrt(3/2)*beta*gamma/k3*a8, 10/3/np.sqrt(6)*(alfa**2 - gamma**2)/k1*a4, - 2*np.sqrt(2/3)*alfa*beta*gamma/k1/k2*a5, 10/3/np.sqrt(6)*(alfa**2 - gamma**2)/k1*a2,
                - 2*np.sqrt(2/3)*alfa*beta*gamma/k1/k2*a3, - (3*alfa**2 + 4*beta**2 + 3*gamma**2)/3/Re, alfa/np.sqrt(6)*(a1 + a9), np.sqrt(3/2)*beta*gamma/k3*(a1+a9), alfa/np.sqrt(6)*a7 + np.sqrt(3/2)*beta*gamma/k3*a8],

               [- alfa/np.sqrt(6)*a6, (gamma**2 - alfa**2)/np.sqrt(6)/k1*a5, alfa*beta*gamma/np.sqrt(6)/k1/k2*a4, alfa*beta*gamma/np.sqrt(6)/k1/k2*a3, (gamma**2 - alfa**2)/np.sqrt(6)/k1*a2,
                - alfa/np.sqrt(6)*(a1 + a9), - k3**2/Re, 0., - alfa/np.sqrt(6)*a6],

               [0., 2/np.sqrt(6)*alfa*beta*gamma/k3/k1*a5, gamma**2*(3*alfa**2 - beta**2 + 3*gamma**2)/np.sqrt(6)/k1/k2/k3*a4, gamma**2*(3*alfa**2 - beta**2 + 3*gamma**2)/np.sqrt(6)/k1/k2/k3*a3,
                2/np.sqrt(6)*alfa*beta*gamma/k3/k1*a2, 0., 0., - k3**2/Re, 0.],

               [0., np.sqrt(3/2)*beta*gamma/k2*a3, np.sqrt(3/2)*beta*gamma/k2*a2, 0., 0., - np.sqrt(3/2)*beta*gamma/k3*a8, 0., - np.sqrt(3/2)*beta*gamma/k3*a6, - 9*beta**2/Re]
               ])
    return J


def MFE_M(q):
    ''' dq/dt and dM/dt '''

    qq     = q[:Ndim].copy()

    J      = MFE_Jac(qq)

    M = q[Ndim:(Ndim+1)*Ndim].reshape((Ndim, Ndim))
    dMdt = np.dot(J, M)

    return np.concatenate([MFE(qq), dMdt.flatten()])

def Gen_MFE(dt, N):

    ''' Compute MFE time series '''

    q   = RK4(q0,dt,N,MFE)

    return q[N_tran+1:]

def Gen_Ch(dt, N, q):

    ''' Compute Lyapunov exponents given a time series and governing equations '''

    qM   = np.zeros((N+1,Ndim*Ndim))

    # time series of M (result of odeint)
    M  = np.zeros((N+1, Ndim, Ndim))

    # time series of Q, R from QR decomposition
    Q = np.zeros((N+1, Ndim, Ndim))
    R = np.zeros((N+1, Ndim, Ndim))

    # Lyapunov exponents history (to check for convergence)
    hl   = np.zeros((N+1, Ndim))

    # initialization of state vector
    qM[0]                   = np.eye(Ndim).flatten()
    M[0]                    = np.eye(Ndim) 
    Q[0]  , R[0]            = gsQR(M[0])

    for i in 1+np.arange(N):
        
        # evolve from T_i-1 -> T_i
        qM[i] = RK4(np.concatenate((q[i-1],qM[i-1])),dt,1,MFE_M)[-1][Ndim:]


        # extract and reshape M matrices from simulation state vector q
        M[i]   = qM[i].copy().reshape((Ndim, Ndim))

        # run QR decomposition
        Q[i],   R[i] = gsQR(M[i])

        # replace M for Q in state vector
        qM[i] = Q[i].flatten()

    #compute evolution of Lyapunov exponents
    for i in 1+np.arange(N):
        hl[i]    = hl[i-1] + np.log(abs(np.diag(R[i])))
    for i in 1+np.arange(N):
        hl[i]   /= (i)*dt

    # lyapunov exponents
    l   = hl[N].copy()

    print(l)

    return hl,l



Ndim = 9
# Problem Definition
Lx = 4*math.pi
Ly = 2.
Lz = 2*math.pi
Re = 400
# Parameter values
alfa  = 2*math.pi/Lx
beta  = math.pi/2
gamma = 2*math.pi/Lz

#initial condition
a10  = 1    
a20  = 0.07066
a30  = -0.07076   
a40  = 0    
a50  = 0   
a60  = 0
a70  = 0  
a80  = 0
a90  = 0   
q00   = np.array([a10, a20, a30, a40, a50, a60, a70, a80, a90]) 

#integration parameters
dt        = .25                                  #timestep
N_tran    = int(200/dt)                          #transient
N         = int(4000/dt) + N_tran                #length of time series
N_t       = 2000                                 #number of time series
q         = np.zeros((N_t,N - N_tran,Ndim))

ii        = 0

#compute time series
for i in range(N_t):
    print(i)
    q0         = q00.copy()
    q0[4]      = a40 + 0.01*np.random.rand()    #each time series starts from a different perturbed point
    q[i]       = Gen_MFE(dt, N)

print('Laminarized precentage', ii/N_t)

ordd = np.nonzero(q[:,-1, 0]) 
q    = q[ordd].copy()

#save time series
fln = './data/MFE_Sri_RK4_dt='+str(dt)+ '_' + str(Re)+'kt=048.h5'
hf = h5py.File(fln,'w')
hf.create_dataset('q',data=q)
hf.close()

#compute Lyapunov exponents
N_t       = 100 #q.shape[0] #number of time series to use for Lyapunov exponents

N         = q.shape[1]
le_hist   = np.zeros((N_t,N,Ndim))
le        = np.zeros((N_t,Ndim))

for i in range(N_t):
    le_hist[i], le[i] = Gen_Ch(dt, N-1, q[i])

print('Lyapunov Exponents', np.mean(le,axis=0))