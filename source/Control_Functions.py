def RK4(q0,dt,N,func,params):
    ''' 4th order RK for autonomous systems described by func '''

    global k_c

    eps_c, n_c = params
    
    q        = np.zeros((N+1,q0.shape[0]))
    q[0]     = q0
    k        = np.zeros(N+1)
    k_c      = 0

    for i in 1+np.arange(N):

        if i <= n_c*downsample:    # times downsample because the dt is a 1/downsample the dt of the ESN 
            k_c = eps_c   # applying control only on the first n_c steps
        else: 
            k_c = 0

        k1   = dt * func(q[i-1])
        k2   = dt * func(q[i-1] + k1/2)
        k3   = dt * func(q[i-1] + k2/2)
        k4   = dt * func(q[i-1] + k3)

        q[i] = q[i-1] + (k1 + 2*k2 + 2*k3 + k4)/6

    return  q


def Gen_Controlled_data(q0):

    ''' Integrate again the equations once a prediction is made '''

    Ndim = 9

    #Control Values
    eps_c     = .0001 #this value is irrelevant, needs to be different than zero and acts as a flag
    
    params    = np.array([eps_c,n_c])

    #Integration Values
    dt        = .25
    N         = int(N_ts/dt) #q is created downsample times longer than the ESN timeseries as the integration happened with dt=0.25
    q         = RK4(q0,dt,N-1,MFE_Control,params) 

    return q[::4]


def MFE_Control(q):
    """
    Defines the differential equations for Moehlis_2004 with changed Reynolds number according to Gen_Controlled_Data
    """
    a1, a2, a3, a4, a5, a6, a7, a8, a9 = q

    # Problem Definition
    Lx = 4*math.pi
    Ly = 2.
    Lz = 2*math.pi
    if k_c != 0:
        Re = 2000
    else:
        Re = Re1
    
    # Parameter values
    alfa  = 2*math.pi/Lx
    beta  = math.pi/2
    gamma = 2*math.pi/Lz


    k1 = np.sqrt(alfa**2 + gamma**2)
    k2 = np.sqrt(gamma**2 + beta**2)
    k3 = np.sqrt(alfa**2 + beta**2 + gamma**2) 

    dqdt = np.array([beta**2/Re * (1. - a1) - np.sqrt(3/2)*beta*gamma/k3*a6*a8 + np.sqrt(3/2)*beta*gamma/k2*a2*a3, # - (k_c)*np.sign(a1),

         - ( 4/3*beta**2 + gamma**2) * a2/Re + 5/3*np.sqrt(2/3)*gamma**2/k1*a4*a6 - gamma**2/np.sqrt(6)/k1*a5*a7 -
         alfa*gamma*beta/np.sqrt(6)/k1/k3*a5*a8 - np.sqrt(3/2)*beta*gamma/k2 * (a1*a3 + a3*a9),# - (k_c)*np.sign(a2),

         - (beta**2 + gamma**2)/Re*a3 + 2/np.sqrt(6)*alfa*beta*gamma/k1/k2 * (a4*a7 + a5*a6) + 
         (beta**2*(3*alfa**2+gamma**2) - 3*gamma**2*(alfa**2+gamma**2))/np.sqrt(6)/k1/k2/k3*a4*a8, # - (k_c)*np.sign(a3),

         - (3*alfa**2 + 4*beta**2)/3/Re*a4 - alfa/np.sqrt(6)*a1*a5 - 10/3/np.sqrt(6)*alfa**2/k1*a2*a6 -
         np.sqrt(3/2)*alfa*beta*gamma/k1/k2*a3*a7 - np.sqrt(3/2)*alfa**2*beta**2/k1/k2/k3*a3*a8 - alfa/np.sqrt(6)*a5*a9, # - (k_c)*np.sign(a4),

         - (alfa**2 + beta**2)/Re*a5 + alfa/np.sqrt(6)*a1*a4 + alfa**2/np.sqrt(6)/k1*a2*a7 - 
         alfa*beta*gamma/np.sqrt(6)/k1/k3*a2*a8 + alfa/np.sqrt(6)*a4*a9 + 2/np.sqrt(6)*alfa*beta*gamma/k1/k2*a3*a6, # - (k_c)*np.sign(a5),

         - (3*alfa**2 + 4*beta**2 + 3*gamma**2)/3/Re*a6 + alfa/np.sqrt(6)*a1*a7 + np.sqrt(3/2)*beta*gamma/k3*a1*a8 +
         10/3/np.sqrt(6)*(alfa**2 - gamma**2)/k1*a2*a4 - 2*np.sqrt(2/3)*alfa*beta*gamma/k1/k2*a3*a5 + alfa/np.sqrt(6)*a7*a9 + np.sqrt(3/2)*beta*gamma/k3*a8*a9, # - (k_c)*np.sign(a6),

         - k3**2/Re*a7 - alfa/np.sqrt(6) * (a1*a6 + a6*a9) + (gamma**2 - alfa**2)/np.sqrt(6)/k1*a2*a5 + alfa*beta*gamma/np.sqrt(6)/k1/k2*a3*a4, # - (k_c)*np.sign(a7),

         - k3**2/Re*a8 + 2/np.sqrt(6)*alfa*beta*gamma/k3/k1*a2*a5 + gamma**2*(3*alfa**2 - beta**2 + 3*gamma**2)/np.sqrt(6)/k1/k2/k3*a3*a4, # - (k_c)*np.sign(a8),

         - 9*beta**2/Re*a9 + np.sqrt(3/2)*beta*gamma/k2*a2*a3 - np.sqrt(3/2)*beta*gamma/k3*a6*a8, # - (k_c)*np.sign(a9)
         ])
    
    return dqdt