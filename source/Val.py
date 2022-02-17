import os
os.environ["OMP_NUM_THREADS"] = '4' # imposes only one core
import numpy as np
import matplotlib.pyplot as plt
import h5py
import skopt
from skopt.space import Real
from skopt.learning import GaussianProcessRegressor as GPR
from skopt.learning.gaussian_process.kernels import Matern, WhiteKernel, Product, ConstantKernel
import matplotlib as mpl
import time
import math
exec(open("Val_Functions.py").read())
exec(open("Functions.py").read())
from skopt.plots import plot_convergence
#Latex
mpl.rc('text', usetex = True)
mpl.rc('font', family = 'serif')


#### Loading data
Ndim      = 9
idx       = range(Ndim)

Re        = 400
t_lyap    = 0.0163**(-1)    # Lyapunov time

downsample  = 4
N_ts        = 10 #only first N_ts used for training and validation

hf       = h5py.File('./data/MFE_Sri_RK4_dt=0.25_'+str(Re)+'kt=048.h5','r')
UU       = np.array(hf.get('q'))[0:N_ts,::downsample] #downsampled every downsample time steps
hf.close()

#checking UU shape
N0 = UU.shape[0]
N1 = UU.shape[1]
print(UU.shape)
U = UU.reshape(N0*N1, Ndim)
print(U.shape)


#### adding noise component-wise to the data

# Set a target SNR in decibel
target_snr_db = 40
sig_avg_watts = np.var(U,axis=0) #signal power
sig_avg_db = 10 * np.log10(sig_avg_watts) #convert in decibel
# Calculate noise then convert to watts
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

#### data management

dt        = 1               # timestep 
N_lyap    = int(t_lyap/dt)  # number of time steps in one Lyapunov time
print(N_lyap, dt)

# number of time steps for washout, train, validation, test
N_washout = N_lyap
N_val     = 2*N_lyap
N_train   = N1 - N_val - N_washout # 196*N_lyap
N_test    = 500*N_lyap

print(N_train/N_lyap)

#compute norm (needed to normalize inputs to ESN)
U_data = U[:N_washout+N_train+N_val]
m = U_data.min(axis=0)
M = U_data.max(axis=0)
norm = M-m

# washout
U_washout = UU[:,:N_washout]
# training
U_t   = UU[:,N_washout:N_washout+N_train-1]
Y_t   = UU[:,N_washout+1:N_washout+N_train]
# validation
Y_v  = UU[:,N_washout+N_train:N_washout+N_train+N_val]
k_v  = 0.5*np.linalg.norm(Y_v[0], axis=1)**2
# training + validation
U_tv  = UU[:,N_washout:N_washout+N_train+N_val-1]
Y_tv  = UU[:,N_washout+1:N_washout+N_train+N_val]


#### ESN Initiliazation Hyperparameters
# To generate the Echo State Networks realizations we set the hyperparameters that we don't optimize with Bayesian Optimization

bias_in = .1 #input bias
bias_out = 1.0 #output bias 
dim = Ndim # dimension of inputs (and outputs) 
connectivity   = 20 

tikh = np.array([1e-5,1e-8,1e-11])  # Tikhonov factor


####  Bayesian Optimization

n_tot = 25    #Total Number of Function Evaluatuions
n_in  = 0     #Number of Initial random points

spec_in     = np.log10(.1)   #range for hyperparameters (spectral radius and input scaling)
spec_end    = np.log10(1)    
in_scal_in  = np.log10(.1)
in_scal_end = np.log10(10)

# In case we want to start from a grid_search, the first n_grid^2 points are from grid search
# if n_grid^2 = n_tot then it is pure grid search
n_grid = 4  # (with n_grid**2 < n_tot you get Bayesian Optimization)

# computing the points in the grid
if n_grid > 0:
    x1    = [[None] * 2 for i in range(n_grid**2)]
    k     = 0
    for i in range(n_grid):
        for j in range(n_grid):
            x1[k] = [spec_in + (spec_end - spec_in)/(n_grid-1)*i,
                     in_scal_in + (in_scal_end - in_scal_in)/(n_grid-1)*j]
            k   += 1

# range for hyperparameters
search_space = [Real(spec_in, spec_end, name='spectral_radius'),
                Real(in_scal_in, in_scal_end, name='input_scaling')]

# ARD 5/2 Matern Kernel with sigma_f in front for the Gaussian Process
kernell = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-1, 3e0))*Matern(length_scale=[0.2,0.2], nu=2.5, length_scale_bounds=(1e-2, 1e1)) 


#Hyperparameter Optimization using either Grid Search or Bayesian Optimization
def g(val):
    
    #Gaussian Process reconstruction
    b_e = GPR(kernel = kernell,
            normalize_y = True, #if true mean assumed to be equal to the average of the obj function data, otherwise =0
            n_restarts_optimizer = 3,  #number of random starts to find the gaussian process hyperparameters
            noise = 1e-10, # only for numerical stability
            random_state = 10) # seed
    
    
    #Bayesian Optimization
    res = skopt.gp_minimize(val,                         # the function to minimize
                      search_space,                      # the bounds on each dimension of x
                      base_estimator       = b_e,        # GP kernel
                      acq_func             = "gp_hedge", # the acquisition function
                      n_calls              = n_tot,      # total number of evaluations of f
                      x0                   = x1,         # Initial grid search points to be evaluated at
                      n_random_starts      = n_in,       # the number of additional random initialization points
                      n_restarts_optimizer = 3,          # number of tries for each acquisition
                      random_state         = 10,         # seed
                           )   
    return res


# ### Validate Echo State
# Select validation function to select the hyperparameters for each realization in the ensemble of networks


#Number of Networks in the ensemble
ensemble = 10
# Which validation strategy (implemented in Val_Functions.ipynb)
N_fo = 30    # number of folds
N_tval = 3   # time series for validation
N_in = 0     # interval before the first fold
N_fw = N_val # how many steps forward the validation interval is shifted 


ti       = time.time()

vals     = [RVC,SSV] # to be able to run multiple validation strategies and multiple sizes of the reservoirs in one go
NN_units = [500,1000]
restart  = [False for i in range(10)]
restart[0] = False  # flags to restart simulation
k_u = 0

for N_units in NN_units:
    
    sparseness =  1 - connectivity/(N_units-1) #set sparseness for different sizes of the reservoir (connectivity is constant)
    
    for val in vals:

        print(val.__name__, N_units) 

        #Quantities to be saved
        # par      = np.zeros((ensemble, 4))      # GP parameters
        # x_iters  = np.zeros((ensemble,n_tot,2)) # coordinates in hp space where f has been evaluated
        # f_iters  = np.zeros((ensemble,n_tot))   # values of f at those coordinates

        tikh_opt = np.zeros(n_tot)
        k        = 0

        # save the final gp reconstruction for each network
        # gps      = [None]*ensemble

        # to save every ensemble and then restart if time is up
        if restart[k_u]:
            fln       = './data/Lor_short_' + str(target_snr_db) + '_' +  val.__name__ + '_' + str(idx) + '_' + str(n_grid) +  '_' + str(N_units) + '3LT_200LT_Multi.h5'
            hf        = h5py.File(fln,'r')
            minimum   = np.array(hf.get('minimum'))
            I         = np.array(hf.get('I'))
            hf.close()
        else:
            minimum  = np.zeros((ensemble, 4))     
            I         = 0

        for i in np.arange(I,ensemble):
            
            k   = 0
            
            print('Realization    :',i+1)
            
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
            
            # Bayesian Optimization
            res        = g(val)
            
            #Saving Quantities for post_processing
            # gps[i]     = res.models[-1]    
            # gp         = gps[i]
            # x_iters[i] = np.array(res.x_iters)
            # f_iters[i] = np.array(res.func_vals)
            minimum[i] = np.append(res.x,[tikh_opt[np.argmin(np.array(res.func_vals))],res.fun])
            # params     = gp.kernel_.get_params()
            # key        = sorted(params)
            # par[i]     = np.array([params[key[2]],params[key[5]][0], params[key[5]][1], gp.noise_])
            
            #Results of the Optimization for each network
            print('Best Results: x', minimum[i,:3], 'f', -minimum[i,-1])

            #saving
            fln = './data/Lor_short_' + str(Re) + '_' + str(target_snr_db) + '_'  + val.__name__ + '_' + str(idx) + '_' + str(n_grid) +  '_' + str(N_units) + '3LT_200LT_Multi.h5'
            hf = h5py.File(fln,'w')
            hf.create_dataset('minimum'   ,data=minimum)
            hf.create_dataset('I'         ,data=i+1)
            hf.close()

    k_u += 1




