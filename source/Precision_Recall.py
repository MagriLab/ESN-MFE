import os
os.environ["OMP_NUM_THREADS"] = "8"
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

kinetic = 0.5*np.linalg.norm(U,axis=1)**2

#### data management

dt        = .25*downsample  # timestep 
N_lyap    = int(t_lyap/dt)  # number of time steps in one Lyapunov time
print(N_lyap, dt)

# number of time steps for washout, train, validation, test
N_washout = N_lyap
N_val     = 2*N_lyap
N_train   = N1 - N_val - N_washout # 196*N_lyap

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
# validation
Y_v  = UU[:N1_val,N_washout+N_train:N_washout+N_train+N_val]
k_v  = 0.5*np.linalg.norm(Y_v[0], axis=1)**2

# ### ESN Initiliazation Parameters

bias_in = .1 #input bias
bias_out = 1.0 #output bias 
dim = Ndim # dimension of inputs (and outputs) 
connectivity   = 20 

# Precision and recall computation

def f_pred(x, Wout, N_pred, N_step, N_in, N_LT):
    """Computes false positives, true positives, etc. for N_LT prediction times separeted by N_step LTs one from the other and starting at N_pred

        For each prediction time, we evaluate if the event is happening in the N_in LTs interval after the PT"""
    
    #hyperparameters
    global rho, sigma_in
    rho      = x[0]
    sigma_in = x[1] 
    
    N_inn    = int(N_in*N_lyap) #size of the window in which we are predicting the event after the Prediction Time and how far into the future we slide
                                #for the next interval
    
    N_inter  = int(N_pred*N_lyap)                          #smallest prediction time
    N_ESN    = N_inter+int(N_step*(N_LT-1)*N_lyap) + N_inn #length of the ESN closed-loop prediction (= to max Prediction Time + N_inn)
    N_data   = N_ESN                                       #change this if the window of the ESN should be different from the true data window
    
    N_testt  = 150 #number of time series we are analysing

    print(N_testt)
    
    #initialize True Negatives, etc. for all the different prediction times
    TN     = np.zeros(N_LT)
    FP     = np.zeros(N_LT)
    FN     = np.zeros(N_LT)
    TP     = np.zeros(N_LT)
    summ   = np.zeros(N_LT)
    
    #Time series in the test set
    for i in range(N1_val,N_testt+N1_val):

        # analyse the part of the time series where prediction is possible
        for k in range((N1-N_washout-N_data-1)//N_lyap):
        
            # run the ESN each N_inn (=1LT)
            p         = N_washout + k*N_inn
            
            #data for washout in each interval
            U_wash    = UU[i,p - N_washout: p]
            
            # if we are not already in an extreme event during washout
            if np.amax(0.5*np.linalg.norm(U_wash,axis=1)**2) < ee:

                #washout for each interval
                xa1    = open_loop(U_wash, np.zeros(N_units), sigma_in, rho)[-1]

                # Prediction
                Yh_t   = closed_loop(N_ESN-1, xa1, Wout, sigma_in, rho)[0]
                
                #Each j is a different prediction time, which scales by N_Steps LTs
                for j in range(N_LT):
                    
                    iii       = N_inter + int(j*N_step*N_lyap) 
                    kinh      = 0.5*np.linalg.norm(Yh_t[iii:iii+N_inn],axis=1)**2
                
                    #data during the prediction window
                    Y_t       = UUU[i,p + iii: p + iii + N_inn] 
                    kin       = 0.5*np.linalg.norm(Y_t,axis=1)**2
                    
                    #maxima of the true data and ESN in the interval after the prediction time
                    kin_max  = np.amax(kin)
                    kinh_max = np.amax(kinh) 

                    # if none crosses ee then is a TN, if only the ESN crosses is a FP 
                    if kin_max < ee or kin[0] > ee:
                        if kinh_max < ee or kinh[0] > ee:
                            TN[j] += 1
                        else:
                            FP[j] += 1

                    # if both cross ee then is a TP, if only the true data crosses is a FN
                    else:
                        if kinh_max > ee and kinh[0] < ee:
                            TP[j] += 1
                        else:
                            FN[j] += 1
                
                    
                
        if (i%5) == 0:
            summ      = TN+FN+FP+TP
            precision = TP/(TP+FP)
            recall    = TP/(TP+FN)
            print('precision',precision)
            print('recall   ',recall)
            print('')
                
            
    return np.column_stack((TN,FN,FP,TP,summ))

print('Precision and Recall Computation')

is_plot   = False

#Compute Ensemble
ensemble = 10
N_LT     = 10 #number of Prediction Times 
ee       = 0.1 #extreme event threshold

#Compute different size of the reservoir and validation strategies in one run
NN_units   = [500] #units in the reservoir
val        = [SSV,RVC]
restart    = [False for i in range(16)]
restart[0] = False #restart flag

k          = 0

for N_units in NN_units:
    
    sparseness =  1 - connectivity/(N_units-1) #setting the correct sparseness for each reservoir

    for jj in val:

        #Validation hyperparameters
        hf       = h5py.File('./data/Lor_short_'+ str(Re) + '_' + str(target_snr_db) + '_'  + jj.__name__ + '_' + str(idx) + '_4_' + str(N_units) +
                             '3LT_200LT_Multi.h5','r')
        Min      = np.array(hf.get('minimum'))
        Min[:,:2] = 10**Min[:,:2] #were computed in log scale
        hf.close()
        print(Min)


        # to save every ensemble and then restart if time is up
        if restart[k]:
            fln       = './data/'+ str(Re) + '_'+ jj.__name__ + '_Prec_Recall_'+ str(target_snr_db) + '_'  + str(N_units) + '.h5'
            hf        = h5py.File(fln,'r')
            temp_pred = np.array(hf.get('temp'))
            I         = np.array(hf.get('I'))
            hf.close()
        else:
            temp_pred = np.zeros((ensemble,N_LT,5))
            I         = 0

        k +=1

        for i in range(I,ensemble):
             
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
            
            #training the network on train+validation data
            rho      = Min[i,0]
            sigma_in = Min[i,1]
            tikh     = Min[i,2]
            Wout     = train(U_washout, U_tv, Y_tv, tikh, sigma_in,rho)[1]
            
            #computing true positives and false negatives            
            N_pred   = .5  #Lyapunov times after which the prediction starts         (minimum prediction time)
            N_interr = 1   #interval length in LT in which the event is predicted
            N_step   = .5  #shift forward in time for different prediction times 

            temp_pred[i] = f_pred(Min[i,:2], Wout, N_pred, N_step, N_interr, N_LT)

            fln = './data/'+ str(Re) + '_'+ jj.__name__ + '_Prec_Recall_'+ str(target_snr_db) + '_'  + str(N_units) + '.h5'
            hf = h5py.File(fln,'w')
            hf.create_dataset('temp'      ,data=temp_pred)
            hf.create_dataset('I'      ,   data=i+1)
            hf.close()