import os
os.environ["OMP_NUM_THREADS"] = "4"
import numpy as np
import matplotlib.pyplot as plt
# plt.style.use('dark_background')
import h5py
import skopt
import scipy as sc
from skopt.space import Real
from skopt.learning import GaussianProcessRegressor as GPR
from skopt.learning.gaussian_process.kernels import Matern, WhiteKernel, Product, ConstantKernel
exec(open("Val_Functions.py").read())
exec(open("Functions_Test.py").read())
import matplotlib as mpl
import math
mpl.rc('text', usetex = True)
mpl.rc('font', family = 'serif')


#### Dataset loading

Ndim      = 9
idx       = range(Ndim)

Re        = 400
t_lyap    = 0.0163**(-1)    # Lyapunov time

downsample  = 4

hf       = h5py.File('./data/MFE_Sri_RK4_dt=0.25_'+str(Re)+'kt=048.h5','r')
UU       = np.array(hf.get('q'))[:,::downsample]
hf.close()


N1_val = 10 #number of time series used during training

N0  = UU.shape[0]
N1  = UU.shape[1]
U   = UU.reshape(N0*N1, Ndim)
UUU = UU.copy()
print(UU.shape, U.shape)


#### Adding noise

# Set a target SNR in decibel
target_snr_db = 40
sig_avg_watts = np.var(U,axis=0) #signal power
sig_avg_db = 10 * np.log10(sig_avg_watts) #convert in decibel
# Calculate noise, then convert to watts
noise_avg_db = sig_avg_db - target_snr_db
noise_avg_watts = 10 ** (noise_avg_db / 10)
# Generate an sample of white noise
mean_noise = 0
noise_volts = np.zeros(U.shape)
seed = 0                        #to be able to recreate the data
rnd  = np.random.RandomState(seed)
for i in range(Ndim):
    noise_volts[:,i] = rnd.normal(mean_noise, np.sqrt(noise_avg_watts[i]),
                                       U.shape[0])
UU  = U + noise_volts

UU  = UU.reshape(N0,N1,Ndim)
kinetic = 0.5*np.linalg.norm(U,axis=1)**2
print(np.mean(kinetic), np.std(kinetic))

#### data management

dt        = .25*downsample  # timestep 
N_lyap    = int(t_lyap/dt)  # number of time steps in one Lyapunov time
print(N_lyap, dt)

# number of time steps for washout, train, validation, test
N_washout = N_lyap
N_val     = 2*N_lyap
N_train   = N1 - N_val - N_washout 

print(N_train/N_lyap)

#compute norm
U_data = U[:N_washout+N_train+N_val]
m = U_data.min(axis=0)
M = U_data.max(axis=0)
norm = M-m

# washout
U_washout = UU[:N1_val,:N_washout]
# training
U_t   = UU[:N1_val,N_washout:N_washout+N_train-1]
Y_t   = UU[:N1_val,N_washout+1:N_washout+N_train]
# training + validation
U_tv  = UU[:N1_val,N_washout:N_washout+N_train+N_val-1]
Y_tv  = UU[:N1_val,N_washout+1:N_washout+N_train+N_val]
NN    = N1_val
k_tv  = 0.5*np.linalg.norm(Y_tv[:NN].reshape(NN*(N_train+N_val-1), Ndim), axis=1)**2
# validation
Y_v  = UU[:N1_val,N_washout+N_train:N_washout+N_train+N_val]


##### Finding the extreme events in the test set

U_test  = UUU[N1_val:].reshape((N0-N1_val*N1),Ndim)
U_test1 = UU[N1_val:].reshape((N0-N1_val*N1),Ndim)
k_test  = 0.5*np.linalg.norm(U_test,axis=1)**2


N_test = 1000*N1
t_ee   = np.zeros(N_test)

ee     = 0.1 #extreme event threshold

for i in range(N_test):
    
    j = int(i/N1)
    if k_test[i] > ee: #flags ok and ok1 set to select only the crossxing of ee and not the timesteps afterwards where k is still larger than ke
        ok = 1
    else:
        ok = 0
        ok1= 1

    #events need to happen at least 10 LTs after time series start, since we start from 10LTs in advance
    #the final condition is needed because we are using a flattened U, where different time series as subsequent one to the other    
    if ok*ok1 and i > (j*N1 + N_lyap*10) and i < ((j+1)*N1): 
        t_ee[i]=1
        ok1    =0
        
indic = np.nonzero(t_ee)[0][:500] #only first 500 to compute PHee


# ### ESN Initiliazation Parameters

bias_in = .1 #input bias
bias_out = 1.0 #output bias 
dim = Ndim # dimension of inputs (and outputs) 
connectivity   = 20 


#### Compute PH for Extreme Events

def f_ee(x, Wout):
    """To see the prediction horizons for the extreme events"""
    
    global rho, sigma_in
    rho      = x[0]
    sigma_in = x[1] 
                
    N_test   = indic.shape[0]   #number of extreme events  
    pred     = np.zeros(N_test) #stores the PHee for each event
    
    #Different Events
    for i in range(N_test):
        
        N_ee     = indic[i] #find extreme event instant in data
        is_pred  = False
        N_pred   = 10*N_lyap #initial value to start the closed loop prediction from

        while not is_pred and N_pred > int(0.5*N_lyap):
        
            # data for washout and target in each interval
            U_wash    = U_test1[N_ee - N_pred - N_washout: N_ee - N_pred]
            Y_t_PH    = U_test[N_ee - N_pred: N_ee + N_lyap] 

            #washout for each interval
            xa1       = open_loop(U_wash, np.zeros(N_units), sigma_in, rho)[-1]

            # Prediction Horizon
            pred[i]   = predictability_horizon_k(xa1,Y_t_PH,Wout,sigma_in, rho) #compute the prediction horizon with respect to absolute value of kin energy
            
            if pred[i] > N_pred/N_lyap:
                is_pred = True
                pred[i] = N_pred/N_lyap #if the prediction gorizon is larger than the interval to the extreme event (tau_e in the paper), then PHee is equal to tau_e
            else:
                N_pred = N_pred - int(0.5*N_lyap)  #if not tau is decreased
                 
        if (i%100) == 0:
            print(i/N_test)
    
    return pred


mean_k    = np.mean(kinetic) #values for the PH with the absolute value of the kinetic energy
threshold = (ee-mean_k)*0.2

is_plot  = False

#multiple validation strategies and size of the reservoir runnable at same time
NN_units   = [500] #units in the reservoir
val        = [SSV,RVC]
restart    = [False for i in range(16)]
restart[0] = False

k=0

for N_units in NN_units:
    
    sparseness =  1 - connectivity/(N_units-1) 

    for jj in val:

        #Bayesian Optimization data
        hf       = h5py.File('./data/Lor_short_'+ str(Re) + '_' + str(target_snr_db) + '_'  + jj.__name__ + '_' + str(idx) + '_4_' + str(N_units) +
                             '3LT_200LT_Multi.h5','r')
        Min      = np.array(hf.get('minimum'))
        Min[:,:2] = 10**Min[:,:2] #input scaling and spectral radius were optimized in log scale
        hf.close()
        print(Min)

        # to save every ensemble and then restart the simulation from where it stopped
        if restart[k]:
            fln       = './data/' + str(Re) + '_' + jj.__name__ + '_PHEE_' + str(target_snr_db) + '_' + str(N_units) + '.h5'
            hf        = h5py.File(fln,'r')
            temp      = np.array(hf.get('PH'))
            I         = np.array(hf.get('I'))
            hf.close()
        else:
            temp    = np.zeros((Min.shape[0],indic.shape[0]))
            I         = 0

        k += 1

        for i in range(I,Min.shape[0]):

            print('Ensemble    :',i+1)

            # Win and W generation
            seed= i+1
            rnd = np.random.RandomState(seed)

            Win = np.zeros((dim+1, N_units))
            for j in range(N_units):
                Win[rnd.randint(0, dim+1),j] = rnd.uniform(-1, 1) #only one element different from zero per row

            # practical way to set the sparseness
            W = rnd.uniform(-1, 1, (N_units, N_units)) * (rnd.rand(N_units, N_units) < (1-sparseness))
            spectral_radius = np.max(np.abs(np.linalg.eigvals(W)))
            W /= spectral_radius #scaled to have unitary spec radius

            #train the network
            rho      = Min[i,0]
            sigma_in = Min[i,1]
            tikh     = Min[i,2]
            Wout     = train(U_washout, U_tv, Y_tv, tikh, sigma_in, rho)[1]

            #compute average PHEE
            temp[i]  = f_ee(Min[i,:2], Wout)
            print(np.median(temp[i]))

            #save results
            fln = './data/' + str(Re) + '_' + jj.__name__ + '_PHEE_' + str(target_snr_db) + '_' + str(N_units) + '.h5'
            hf = h5py.File(fln,'w')
            hf.create_dataset('PH'   ,data=temp)
            hf.create_dataset('I'      ,   data=i+1)
            hf.close()
