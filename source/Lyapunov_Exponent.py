import numpy as np
import scipy as sc
from scipy.integrate import odeint
from matplotlib.pyplot import *
from matplotlib import rcParams
import matplotlib as mpl
import scipy.stats
import math
import h5py

mpl.rc('text', usetex = True)
mpl.rc('font', family = 'serif')

rcParams['axes.labelsize'] = 12
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['legend.fontsize'] = 12
rcParams['font.size'] = 7

def RK4(q0,dt,N,func):
    ''' 4th order RK for autonomous systems described by func '''

    q        = np.zeros((N+1,q0.shape[0]))
    q[0]     = q0

    for i in 1+np.arange(N):

        k1   = dt * func(q[i-1])
        k2   = dt * func(q[i-1] + k1/2)
        k3   = dt * func(q[i-1] + k2/2)
        k4   = dt * func(q[i-1] + k3)

        q[i] = q[i-1] + (k1 + 2*k2 + 2*k3 + k4)/6

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

# Problem Definition
Ndim = 9
Lx = 4*math.pi
Ly = 2.
Lz = 2*math.pi

Re = 400
print(Re)

alfa  = 2*math.pi/Lx
beta  = math.pi/2
gamma = 2*math.pi/Lz

# Integration parameters
dt       = .25     #timestep
N_points = 50      #number of points along the attractor
N        = 5000    #length of each time series of the perturbations (length should assure nonlinear saturation of the norm)
t        = dt*np.arange(N)

#load MFE time series
fln = './data/MFE_Sri_RK4_dt='+str(dt)+ '_' + str(Re)+'kt=048.h5'
hf = h5py.File(fln,'r')
q = np.array(np.array(hf.get('q')))
hf.close()


#initialize
error    = np.zeros((N_points, N))
le       = np.zeros(N_points)

ioff() # Comment out this line and uncomment show() to see it real time
fig = figure(figsize=(5,3))

#compute the perturbed trajectory and compare it to the unperturbed one 
for i in range(N_points):
    x0P = q[i,500].copy() + 1e-6  #perturbed initial condition
    x_t  =  q[i,500:500+N].copy() #unperturbed trajectory
    x_tP =  RK4(x0P,dt,N-1,MFE)   #perturbed trajectory

    error[i] = np.linalg.norm(np.abs(x_t-x_tP),axis=1)

    cut      = np.argmax(np.log10(error[i])>-1) #nonlinear saturation happens when error is equal to 0.1 (system-dependent)
    if cut == 0: cut = -1                       #if there's no nonlinear saturation
    
    # interpolate the evolution of the perturbation with a line in log scale
    P        = np.polyfit(t[:cut],np.log(error[i,:cut]),1)
    f        = np.poly1d(P)
    tFit     = np.arange(0,cut*dt,dt)
    errorFit = f(tFit)
    le[i] = P[0]  #polyfit returns the polynomial coefficents and the first one for degree=1 is the slope #*1
    print('Lyapunov Exponent from the ' + str(i) + '-th point: %f' % P[0])


# Plotting all the lines describing the error from the different points

    plot(t,np.log10(error[i]), color='k',linewidth=1, alpha=.1)
    plot(tFit,errorFit*np.log10(np.e),'blue',linewidth=1,alpha=0.2)
    axhline(-1,color='k',linestyle='-') #threshold for saturation
    print(i)

print('Lyapunov exponent:', np.mean(le))

minorticks_on()
xlabel('Time')
ylabel(r'$\log_{10}|| \delta\phi ||$',fontsize=18)
tight_layout()
savefig('Figures/Perturbations_Evolution.pdf')
close()


# Plot Lyapunov exponent convergence and histogram from different points
ioff() # Comment out this line and uncomment show() to see it real time
fig = figure(figsize=(8,3))
#LE convergence
subplot(121)
Meannn = np.zeros(N_points)

for i in range(N_points):
    Meannn[i] = np.mean(le[:i+1])
plot(Meannn)
#LE from different point histogram
axx=subplot(122)
axx.yaxis.set_label_position("right")
axx.yaxis.tick_right()
hist(le,bins=1+N_points//10,density=True)
tight_layout()
savefig('Figures/LE_convergence.pdf')
close()
