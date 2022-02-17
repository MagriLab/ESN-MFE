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
exec(open("Control_Functions.py").read())
import matplotlib as mpl
import math
import time
mpl.rc('text', usetex = True)
mpl.rc('font', family = 'serif')


#### Dataset loading

Ndim      = 9
idx       = range(Ndim)

Re1       = 400
t_lyap    = 0.0163**(-1)    # Lyapunov time

downsample  = 4

hf       = h5py.File('./data/MFE_Sri_RK4_dt=0.25_'+str(Re1)+'kt=048.h5','r')
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
# validation
Y_v  = UU[:N1_val,N_washout+N_train:N_washout+N_train+N_val]
k_v  = 0.5*np.linalg.norm(Y_v[0], axis=1)**2

# ## Finding the extreme events in the time series

# In[6]:


# k_test = kinetic[N_washout+N_train+N_val:].copy()
# U_test = UUU[N_washout+N_train+N_val:].copy()

# # k_test = k_test[:k_test.shape[0]//10]

# N_test = k_test.shape[0]
# print(N_test)
# t_ee   = np.zeros(N_test)

ee     = 0.1

# for i in range(1,N_test-1):
    
#     j = int(i/N1)
# #     print(i,j)
#     if k_test[i] > ee:
#         ok = 1
#     else:
#         ok = 0
#         ok1= 1
        

#     if ok*ok1 and i > (j*N1 + N_washout) and i < ((j+1)*N1):
# #         print(i, (j*N1 + N_washout),  ((j+1)*N1))
#         t_ee[i]=1
#         ok1    =0
        
# indic = np.nonzero(t_ee)[0]
# indic = indic[:]
# print(indic.size)


# ### Import Results From  Optimization Runs

# In[11]:


# N_units = 1000 #units in the reservoir

# #BO and Grid Search in KFC
# hf       = h5py.File('./data/Lor_short_RVC_' + str(idx) + '_4_' + str(N_units) +
#                      '3LT_200LT_Multi.h5','r')
# Min      = np.array(hf.get('minimum'))
# Min[:,:2] = 10**Min[:,:2]
# hf.close()
# print(Min)


# ### ESN Initiliazation Parameters

# In[12]:


bias_in = .1 #input bias
bias_out = 1.0 #output bias 
dim = Ndim # dimension of inputs (and outputs) 
connectivity   = 20 

# sparseness =  1 - connectivity/(N_units-1) 


# In[17]:


# Control through the precision and recall framework

def f_pred(x, Wout, N_pred, N_step, N_in, N_LT, N_c, N_contr, N_ts, k_nl):
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
    
    
    #initialize True Negatives, etc. for all the different prediction times
    TN     = np.zeros(N_LT)
    FP     = np.zeros(N_LT)
    FN     = np.zeros(N_LT)
    TP     = np.zeros(N_LT)
    summ   = np.zeros(N_LT)

    UU_c  = UU[N1_val:N1_val+N_contr, :N_ts].copy()
    U_c   = UUU[N1_val:N1_val+N_contr, :N_ts].copy() #if there is no extreme event after the control for N_ts then we use a new uncontrolled time series

    k_skip = 100 #flag to identify event happening at begging of time series


    # check if in the first part of the time series there is an extreme event (from the original data, when no )
    for i  in range(N_contr):
        if (0.5*np.linalg.norm(U_c[i,:N_washout + N_inter + N_c*N_lyap],axis=1)**2).max() > ee:
            U_c[i]     = 0
            U_c[i,0,0] = k_skip


    for i in range(N_contr-1): #N_contr is how many times we perform control

        pred_c = False #when it becomes true, control is performed and we generate a new time series

        # skip original data with extreme events that are too early in the time series to be predicted
        if U_c[i,0,0] == k_skip:
            continue

        #if the event is not dealyed by enough after control to be predicted again after control is performed, we have predicted the event but not successfully suppressed it    
        if (0.5*np.linalg.norm(U_c[i,:N_washout + N_inter + N_c*N_lyap],axis=1)**2).max() > ee: 
            print(i,'Not delayed Event')


        # looping inside each time series
        for k in range((N_ts-N_washout-(N_pred+N_c)*N_lyap)//N_lyap):
            
            if pred_c: break
            
            # run the ESN each N_inn (=1LT), starting after also control is applied (rounded down to the lower LT)
            p         = N_washout + N_c*N_lyap + k*N_inn #int(N_in*N_lyap)
            
            #data for washout in each interval (with noise)
            U_wash    = UU_c[i,p - N_washout: p].copy()

            # if we are not already in an extreme event
            if np.amax(0.5*np.linalg.norm(U_wash,axis=1)**2) < ee:

                # perform washout for the current interval
                xa1    = open_loop(U_wash, np.zeros(N_units), sigma_in, rho)[-1]

                # Prediction (for a long interval and then when evaluate all the smaller intervals)
                Yh_t   = closed_loop(N_ESN-1, xa1, Wout, sigma_in, rho)[0]
                kin_t  = 0.5*np.linalg.norm(Yh_t,axis=1)**2

                
                # Prediction for different intervals in the future
                for j in range(N_LT): #N_LT = 1 in control, as we analyze only a fixed prediction time
                    
                    iii       = N_inter + int(j*N_step*N_lyap)
                    kinh      = kin_t[iii:iii+N_inn].copy() #kinetic energy in the interval
                
                    #data during the prediction window
                    Y_t       = U_c[i,p + iii: p + iii + N_inn].copy()
                    kin       = 0.5*np.linalg.norm(Y_t,axis=1)**2
                    
                    #maxima
                    kin_max  = np.amax(kin)
                    kinh_max = np.amax(kinh) 

                    if kin_max < ee or kin[0] > ee:       #if in the data an event is not happening (because either the max_value is too small or the initial value is already extreme)
                        if kinh_max < ee or kinh[0] > ee: #if the same is true for the ESN prediction
                            TN[j] += 1
                        else: #if the same is not true for the ESN prediction
                            FP[j] += 1

                            #the network is predicting an event, which means that the control strategy is actived
                            #storing the new data as the next time series (i+1), monitoring the system starts from ther
                            U_c[i+1] = Gen_Controlled_data(U_c[i,p + iii - int(k_nl*N_lyap)]) #generates controlled data from k_nl before the predicted ee
                            
                            #adding the noise to the new time series
                            noise_volts = np.zeros((U_c.shape[1],U_c.shape[2]))
                            for kk in range(Ndim):
                                noise_volts[:,kk] = rnd.normal(0, np.sqrt(noise_avg_watts[kk]), U_c.shape[1])
                            UU_c[i+1] = U_c[i+1] + noise_volts
                            
                            pred_c = True #go to next time series

                            U_c[i,p+iii-int(k_nl*N_lyap):] = 0 #putting equal to zero everything after control starts in the current time series
                            

                    else: #if in the data there is an extreme event
                        if kinh_max > ee and kinh[0] < ee: #if also in the ESN there is one (because the max value is larger than the threshold and the initial value smaller)
                            TP[j] += 1
                            
                            #as done for the False Positive above
                            U_c[i+1] = Gen_Controlled_data(U_c[i,p + iii - int(k_nl*N_lyap)]) #generates controlled data from k_nl before the predicted ee
                                                                                        
                            noise_volts = np.zeros((U_c.shape[1],U_c.shape[2]))
                            for kk in range(Ndim):
                                noise_volts[:,kk] = rnd.normal(0, np.sqrt(noise_avg_watts[kk]), U_c.shape[1])
                            UU_c[i+1] = U_c[i+1] + noise_volts
                            
                            pred_c = True

                            # if is_plot:

                            #     NN_fut = N_ESN1 + N_pred #7 # how long in the future to plot the controlled and uncontrolled

                            #     plt.rcParams["figure.figsize"] = (10,5)
                            #     plt.rcParams["font.size"] = 25

                            #     plt.figure()
                            #     plt.axvline(p/N_lyap+N_pred, c='k', alpha=0.3)
                            #     plt.axvline(p/N_lyap+N_pred + N_data1, c='k', alpha=0.3)
                            #     plt.axhline(.1, c='k', alpha=0.3)

                            #     plt.ylim(0,.3)

                            #     plt.plot(p/N_lyap + np.arange(U_c[i,p: p + NN_fut*N_lyap].shape[0])/N_lyap,
                            #         0.5*np.linalg.norm(U_c[i,p: p + NN_fut*N_lyap],axis=1)**2, label='Uncontrolled', c='k', linewidth=3)

                            #     plt.plot(p/N_lyap + np.arange(N_ESN)/N_lyap,
                            #         0.5*np.linalg.norm(Yh_t,axis=1)**2, label='ESN', linestyle='--', c='chocolate', linewidth=3)

                            #     plt.plot(p/N_lyap + np.arange(iii-int(k_nl*N_lyap),(NN_fut)*N_lyap)/N_lyap,
                            #         0.5*np.linalg.norm(U_c[i+1,:(NN_fut*N_lyap + int(k_nl*N_lyap)) - iii],axis=1)**2, label='Suppressed', c='royalblue', linewidth=3)


                            #     plt.legend()
                            #     plt.xlabel('Time [LT]')
                            #     plt.ylabel('$k$')
                            #     plt.tight_layout(pad=0.2)

                            #     plt.show()


                            U_c[i,p+iii-int(k_nl*N_lyap):] = 0 #putting equal to zero everything after control start to be applied as the continuing of the time series is saved in the i+i
                           
                        else:
                            FN[j] += 1


                            # if is_plot:

                            #     NN_fut = N_ESN1 + N_pred #7 # how long in the future to plot the controlled and uncontrolled

                            #     plt.rcParams["figure.figsize"] = (10,5)
                            #     plt.rcParams["font.size"] = 25

                            #     plt.figure()
                            #     plt.axvline(p/N_lyap+N_pred, c='k', alpha=0.3)
                            #     plt.axvline(p/N_lyap+N_pred + N_data1, c='k', alpha=0.3)
                            #     plt.axhline(.1, c='k', alpha=0.3)

                            #     plt.ylim(0,.3)

                            #     plt.plot(p/N_lyap + np.arange(U_c[i,p: p + NN_fut*N_lyap].shape[0])/N_lyap,
                            #         0.5*np.linalg.norm(U_c[i,p: p + NN_fut*N_lyap],axis=1)**2, label='Uncontrolled', c='k', linewidth=3)

                            #     plt.plot(p/N_lyap + np.arange(N_ESN)/N_lyap,
                            #         0.5*np.linalg.norm(Yh_t,axis=1)**2, label='ESN', linestyle='--', c='chocolate', linewidth=3)

                            #     # plt.plot(p/N_lyap + np.arange(iii-int(k_nl*N_lyap),(NN_fut)*N_lyap)/N_lyap,
                            #     #     0.5*np.linalg.norm(U_c[i+1,:(NN_fut*N_lyap + int(k_nl*N_lyap)) - iii],axis=1)**2, label='Suppressed', c='royalblue', linewidth=3)


                            #     plt.legend()
                            #     plt.xlabel('Time [LT]')
                            #     plt.ylabel('$k$')
                            #     plt.tight_layout(pad=0.2)

                            #     plt.show()


                            print('FN')             
            
    return U_c[:N_contr-1]



print('Control Starting')

is_plot   = False

#Compute Ensemble
ensemble   = 10
N_LT       = 1   # only 1 Prediction Time at a time
N_pred     = 2   # Prediction Time
# N_data   = 1  #Lyapunov times during which the prediction takes place
# N_ESN    = 5
N_interr = 1           #interval length in LT in which the event is predicted
N_step   = .5          # (irrelevant value as in control we are doing only 1LT at a time)         
k_nl     = 1           # we act k_nl LTs before the predicted extreme event
nn_c     = 1.5         # the control strategy lasts nn_c LTs
n_c      = nn_c*N_lyap 


#To run multiple reservoir sizes and control strategies
NN_units   = [500] #units in the reservoir
val        = [RVC]
restart    = [False for i in range(16)]
restart[0] = False #flag to restart simulation from where it was interrupted
k          = 0

N_ts       = 20*N_lyap    # length of the time series computed when the control strategy is activated (faster as you do not compute the entire 60 LTs only to discard them when a new event is predicted)
N_contr    = 1000         # Number of times we perform control
rangee     = [0.012,0.48] # range for the histogram of the controlled system that we save. It eliminated the laminarized time series and the ones put to zero  
N_bins     = 100          # number of bins in the histogram


for N_units in NN_units:
    
    sparseness =  1 - connectivity/(N_units-1) #setting the correct sparseness for each reservoir

    for jj in val:

        #Validation hyperparameters
        hf       = h5py.File('./data/Lor_short_'+ str(Re1) + '_' + str(target_snr_db) + '_'  + jj.__name__ + '_' + str(idx) + '_4_' + str(N_units) +
                             '3LT_200LT_Multi.h5','r')
        Min      = np.array(hf.get('minimum'))
        Min[:,:2] = 10**Min[:,:2] #were computed in log scale
        hf.close()
        print(Min)

        ## to save every ensemble and then restart if time is up
        if restart[k]:
            fln       = './data/'+ str(Re1) + '_' + jj.__name__ + '_New_Control_'+ str(N_pred) + '_' + str(N_units) +'.h5'
            hf        = h5py.File(fln,'r')
            temp_pred = np.array(hf.get('temp'))
            I         = np.array(hf.get('I'))
            hf.close()
        else:
            temp_pred = np.zeros((ensemble,N_bins))
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
            
            # training the network
            rho      = Min[i,0]
            sigma_in = Min[i,1]
            tikh     = Min[i,2]
            Wout     = train(U_washout, U_tv, Y_tv, tikh, sigma_in,rho)[1]

            # Controlling the system and getting the controlled time series            
            temp_predd = f_pred(Min[i,:2], Wout, N_pred, N_step, N_interr, N_LT, int(nn_c), N_contr, N_ts, k_nl)

            print(temp_pred[i].shape)

            # put to zero the laminarized time series
            for j in range(N_contr-1):
                if temp_predd[j,:,0].max() > 0.99:
                    temp_predd[j] = 0

            #computing the histogram from the suppressed time series to be saved        
            a1, a2 = np.histogram((0.5*np.linalg.norm(temp_predd,axis=2)**2).flatten(), range=(rangee[0],rangee[1]), bins=N_bins, density=True)
            temp_pred[i] = a1.copy()

            # if is_plot:
            #     b1, b2 = np.histogram((0.5*np.linalg.norm(UUU[N1_val:N1_val+N_contr, :N_ts],axis=2)**2).flatten(), range=(1e-2,.48), bins=N_bins, density=True)

            #     plt.rcParams["figure.figsize"] = (10,5)
            #     plt.rcParams["font.size"] = 25

            #     plt.figure()
            #     plt.yscale('log')
            #     plt.plot(a2[:-1] + (a2[1]-a2[0])/2, a1, label='Suppressed', linewidth=3, color='royalblue')
            #     plt.plot(b2[:-1] + (b2[1]-b2[0])/2, b1, label='Unsupressed', linewidth=3, color='k')
            #     plt.ylabel('PDF')
            #     plt.xlabel('$k$')
            #     plt.legend()
            #     plt.tight_layout(pad=0.1)
            #     plt.savefig('New_Control' + '_' + str(N_units) + '_RVC.pdf')
            #     plt.show()

            #saving
            fln = './data/'+ str(Re1) + '_' + jj.__name__ + '_New_Control_'+ str(N_pred) + '_'  + str(N_units) +'.h5'
            hf = h5py.File(fln,'w')
            hf.create_dataset('temp'      ,data=temp_pred)
            hf.create_dataset('Parameters'      ,data=np.array([rangee[0],rangee[1],N_bins]))
            hf.create_dataset('I'      ,   data=i+1)
            hf.close()


# In[ ]:


# print(temp_pred)

# fln = './data/RVC_Prec_Recall_'+  str(N_units) +'.h5'
# hf = h5py.File(fln,'w')
# hf.create_dataset('temp'      ,data=temp_pred)
# hf.close()

# for i in range(ensemble):
#     TN,FN,FP,TP,summ = temp_pred[i,3]
#     precision = TP/(TP+FP)
#     recall    = TP/(TP+FN)
#     F_1       = 2/(1/recall+1/precision)
#     print(precision, recall, F_1)
