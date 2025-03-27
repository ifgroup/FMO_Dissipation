# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 13:09:17 2025

@author: ignac
"""
import numpy as np

#       DEFINING THE CONSTANTS: hbar and kb are set to be 1                 #  
HBAR_MKS = 1.0545718e-34
kb_MKS = 1.38064852e-23
C_CGS = 2.997992458e+10
PS_TO_S = 1.0e-12
FS_TO_S = 1.0e-15
TWOPI = (2.0 * np.pi)
TUNIT_TO_S = (HBAR_MKS / kb_MKS)
TUNIT_TO_FS  =(TUNIT_TO_S / FS_TO_S)
TUNIT_TO_PS  =(TUNIT_TO_S / PS_TO_S)
WAVENO_TO_EUNIT  =(C_CGS * TWOPI * TUNIT_TO_S)
WAVENO_TO_OMEGA_FS = (C_CGS * TWOPI * FS_TO_S)
Jdis_to_ps = WAVENO_TO_OMEGA_FS**2/(WAVENO_TO_EUNIT**2 ) *TUNIT_TO_FS *1000

#                               FUNCTIONS                                  #
# RUNGE-KUTTA INTEGRATOR
def RK4(f,xi,yi,h):                           
    k1 = f(xi, yi)                          
    k2 = f(xi + (h/2.0), yi + ((k1*h)/2.0))     
    k3 = f(xi + (h/2.0), yi + ((k2*h)/2.0))
    k4 = f(xi + h, yi + (k3*h))
    y = yi + ((h/6.0) * (k1 + (2.0*k2) + (2.0*k3) + k4))
    # print(y.shape)
    return y

def gaussian(x, mean, std, amplitude):
    return amplitude * np.exp(-0.5*((x - mean) / std) ** 2)

# EVOLUTION OPERATOR
def evol_generator(t, rho):
    matrix = np.zeros_like(rho)
    for i in range(len(rho)):
        matrix[i] = -rho[i] * np.sum(K, axis=0)[i] + np.sum([rho[j] * K[i][j] for j in range(len(rho)) if j != i])
    return matrix

# LINE-BROADENING FUNCTION
def line_broadening(t,site):
    arg = w_axis*t
    dummy_real = np.sum(J[site] * wfac * (1.0 - np.cos(arg))/w_axis)
    dummy_imag = np.sum(J[site] *  (np.sin(arg) - arg) /w_axis)
    dum= dummy_real + 1.0j*dummy_imag
    return dum
    
# THERMAL-FACTOR
def thermal_factor(Temperature,w_axis):
    return 1 / np.tanh(w_axis/(2.0*Temperature))

def fluorescence(site):
    lamb =  np.sum(J[site])       # NUMERICAL TOTAL REORGANIZATION ENERGY
    cent = -E[site]+lamb
    coeff = np.exp(-g[site].real)
    arg = cent * time_axis + g[site].imag
    return coeff * np.cos(arg) + 1.0j *coeff * np.sin(arg)

def absorption(site):
    lamb =  np.sum(J[site])       # NUMERICAL TOTAL REORGANIZATION ENERGY
    coeff = np.exp(-g[site].real)
    cent = -(E[site]+lamb)
    arg = cent *time_axis- g[site].imag
    return coeff * np.cos(arg) +1.0j*coeff * np.sin(arg)

def rate_constant(i,j):
    return 2.0 * V2[i][j] * np.real(np.sum(FA[i][j])*dt - (FA[i][j][0]+FA[i][j][-1])*dt*0.5)

def dissipative_potential(i,j,k):
    term1=np.cos(w_axis[k]*time_axis)
    term2=-1.0j*wfac[k]*np.sin(w_axis[k]*time_axis)
    dummy= np.real(FA[j][i]* (term1+term2))
    return np.sum(dummy)*dt - (dummy[0]+dummy[-1])*dt/2


# SIMULATION PARAMETERS
System_Dim=7                                        # NUMBER OF SITES
T = 300.0                                       # TEMPERATURE

# FREQUENCY GRID AND DISCRETE SPECTRAL DENSITY FOR EACH SITE
b=np.loadtxt('Spectral_Density/spd_300K_exc1.txt')
w_axis=b[:,0]*WAVENO_TO_EUNIT
dw= (w_axis[-1]-w_axis[0])/ float(len(w_axis))
nmode = len(w_axis)

sp=np.zeros([System_Dim,nmode],dtype=float)
for i in range(1,System_Dim+1):
    b=np.loadtxt('Spectral_Density/spd_300K_exc'+str(i)+'.txt')
    sp[i-1]=b[:,1]*WAVENO_TO_EUNIT

# DISCRETE REORGANIZATION ENERGY FOR EACH SITE
J=np.zeros([System_Dim,nmode],dtype=float)
J_s=np.zeros([System_Dim,nmode],dtype=float)
Jfast=np.zeros([System_Dim,nmode],dtype=float)
Jslow=np.zeros([System_Dim,nmode],dtype=float)
ck = np.zeros([System_Dim,nmode],dtype=float)
Sw=np.zeros([System_Dim,nmode],dtype=float)
wstar=np.zeros(System_Dim,dtype=float)
for i in range(System_Dim):
    wstar[i] = 20.0*WAVENO_TO_EUNIT


for j in range(System_Dim):
    for i in range(nmode):
        if w_axis[i] < wstar[j]:
            Sw[j][i] = (1 - (w_axis[i]/wstar[j])**2 )**2

for i in range(System_Dim):        
    Jslow[i]= Sw[i]*sp[i]
    Jfast[i]= (1-Sw[i])*sp[i]

for j in range(System_Dim):
    J[j] = Jfast[j] / w_axis * dw
    J_s[j] = Jslow[j] / w_axis * dw

#THERMAL FACTOR 
wfac = np.zeros([nmode],dtype=float)
wfac = thermal_factor(T, w_axis)


V2=np.zeros([System_Dim,System_Dim],dtype=float)        # ELECTRONIC COUPLING SQUARE IN CM-1
V2[0][1]=87.7
V2[1][0]=87.7
V2[0][2]=5.5
V2[2][0]=5.5
V2[0][3]=5.9
V2[3][0]=5.9
V2[0][4]=6.7
V2[4][0]=6.7
V2[0][5]=13.7
V2[5][0]=13.7
V2[0][6]=9.9
V2[6][0]=9.9
V2[1][2]=30.8
V2[2][1]=30.8
V2[1][3]=8.2
V2[3][1]=8.2
V2[1][4]=0.7
V2[4][1]=0.7
V2[1][5]=11.8
V2[5][1]=11.8
V2[1][6]=4.3
V2[6][1]=4.3
V2[2][3]=53.5
V2[3][2]=53.5
V2[2][4]=2.2
V2[4][2]=2.2
V2[2][5]=9.6
V2[5][2]=9.6
V2[2][6]=6.0
V2[6][2]=6.0
V2[3][4]=70.7
V2[4][3]=70.7
V2[3][5]=17
V2[5][3]=17
V2[3][6]=63.3
V2[6][3]=63.3
V2[4][5]=81.1
V2[5][4]=81.1
V2[4][6]=1.3
V2[6][4]=1.3
V2[5][6]=39.7
V2[6][5]=39.7
V2=(V2*WAVENO_TO_EUNIT)**2

 
# TIME GRID
dt = 0.5 / TUNIT_TO_FS
tmax = 30000.0 / TUNIT_TO_FS
ti=0.0 / TUNIT_TO_FS
ntgrid = int(tmax/dt)
time_axis=np.zeros([ntgrid],dtype=float)
for i in range(ntgrid):
    time_axis[i]=ti+float(i)*dt


# LINE-BROADENING FUNCTION
g=np.zeros([System_Dim,ntgrid],dtype=complex)
for i in range(System_Dim):
    for j in range(ntgrid):
        g[i][j]= line_broadening(time_axis[j],i)
    

traj=500
popavg = np.zeros([System_Dim,ntgrid],dtype=float)
Diss_avg = np.zeros([System_Dim,nmode],dtype=float)
Diss_time_avg = np.zeros([System_Dim,ntgrid],dtype=float)
Diss_Tavg = np.zeros([nmode],dtype=float)
Diss_time_avera = np.zeros([System_Dim, int(ntgrid/24.0), nmode],dtype=float)

for z in range(1,traj+1):
    
    
    E = np.zeros([System_Dim],dtype=float)              # SITE ENERGIES IN CM-1
    E[0]=(410)*WAVENO_TO_EUNIT + np.random.normal(0,  7.23915584e+01)
    E[1]=(530)*WAVENO_TO_EUNIT + np.random.normal(0,  7.76493066e+01)
    E[2]=(210)*WAVENO_TO_EUNIT + np.random.normal(0,  8.74829551e+01)
    E[3]=(320)*WAVENO_TO_EUNIT + np.random.normal(0,  5.13076827e+01)
    E[4]=(480)*WAVENO_TO_EUNIT + np.random.normal(0,  5.80267079e+01)
    E[5]=(630)*WAVENO_TO_EUNIT + np.random.normal(0,  5.93813972e+01)
    E[6]=(440)*WAVENO_TO_EUNIT + np.random.normal(0,  6.60604728e+01)
    
    
    # FLUORESCENCE AND ABSORPTION FUNCTIONS
    F=np.zeros([System_Dim,ntgrid],dtype=complex)
    A=np.zeros([System_Dim,ntgrid],dtype=complex)
    for i in range(System_Dim):
        F[i] = fluorescence(int(i))
        A[i] = absorption(int(i))
    
    # FLUORESCENCE/ABSORPTION OVERLAP
    FA=np.zeros([System_Dim,System_Dim,ntgrid],dtype=complex)
    for i in range(System_Dim):
        for j in range(System_Dim):
            FA[i][j]= np.conjugate(F[i])*A[j] 
          
    # POPULATION RATE CONSTANT
    K = np.zeros([System_Dim,System_Dim],dtype=float)
    for i in range(System_Dim):
        for j in range(System_Dim):
            K[j][i] = rate_constant(int(i),int(j))
    
              
    # DISSIPATIVE POTENTIAL
    I=np.zeros([System_Dim,System_Dim, nmode],dtype=float)
    for i in range(System_Dim):
        for j in range(System_Dim):
            if i != j:
                for k in range(nmode):
                    I[i][j][k]= dissipative_potential(int(i),int(j),int(k))

    # DISSIPATIVE SPECTRAL DENSITY
    Jdis=np.zeros([System_Dim,System_Dim,System_Dim, nmode],dtype=float)
    for k in range(System_Dim):
        for i in range(System_Dim):
            for j in range(System_Dim):
                if k == i or k == j:
                    Jdis[k][i][j]= 2 * V2[i][j] * I[i][j] * J[k]

    # POPULATION DYNAMICS  
    pop = np.zeros([System_Dim, ntgrid])
    rho = np.zeros(int(System_Dim))
    rho[0] = 1
    for i in range(ntgrid):
        t = time_axis[i]
        for j in range(System_Dim):
            pop[j][i] = rho[j]
        dummy = RK4(evol_generator, t, rho, dt)
        rho = dummy

    
    # SITE DISSIPATION AND TOTAL DISSIPATION  as a function of frequency
    Int_pop=np.zeros([System_Dim],dtype=float)
    for i in range(System_Dim):
            Int_pop[i]= np.sum(pop[i])*dt - (pop[i][0]+pop[i][-1])*0.5*dt
    
    Diss=np.zeros([System_Dim, nmode],dtype=float)       
    for i in range(System_Dim):
        for j in range(System_Dim):
            if j!=i:
                Diss[i] += Jdis[i][j][i] * Int_pop[i] + Jdis[i][i][j] * Int_pop[j]
            else:
                pass
    Diss_total=np.zeros([nmode],dtype=float)
    for i in range(System_Dim):
        Diss_total += Diss[i]
                
    # TIME DEPENDENT DISSIPATION IN A FREQUENCY RANGE
    somet = np.zeros([ntgrid],dtype=float)
    wi=w_axis[0]   #in cm-1
    wf=w_axis[-1]  #in cm-1
    j_values = []
    for j in range(nmode):
        if wi < w_axis[j] / WAVENO_TO_EUNIT < wf:
            j_values.append(j)
    
    sumj=np.zeros([System_Dim,System_Dim,System_Dim],dtype=float)
    for k in range(System_Dim):
        for i in range(System_Dim):
            for j in range(System_Dim):
                if i != j:
                    sumj[k][i][j] = np.sum(Jdis[k][i][j][j_values[0]:j_values[-1]])*dw - (Jdis[k][i][j][j_values[0]]+Jdis[k][i][j][j_values[-1]])*0.5*dw 
    
    Disst=np.zeros([System_Dim,ntgrid],dtype=float)
    intpop=np.zeros(System_Dim,dtype=float)        
    for i in range(ntgrid):
        for j in range(System_Dim):
            intpop[j] = np.sum(pop[j][0:i])*dt-(pop[j][0]+pop[j][i])*dt*0.5
        for h in range(System_Dim):
            for k in range(System_Dim):
                  if h!=k:
                      Disst[h][i] += sumj[h][k][h] * intpop[h] + sumj[h][h][k] * intpop[k]
    
    for i in range(System_Dim):
        popavg[i] += pop[i]
        Diss_avg[i] += Diss[i]
        Diss_time_avg[i] += Disst[i]

    Diss_Tavg += Diss_total
    
    Diss_time=np.zeros([System_Dim, int(ntgrid/24.0), nmode],dtype=float)
    for i in range(int(ntgrid/3)):
        intpop=np.zeros(System_Dim,dtype=float)
        if i%8==0:
            for j in range(System_Dim):
                intpop[j] = np.sum(pop[j][0:i])*dt-(pop[j][0]+pop[j][i])*dt*0.5
            for h in range(System_Dim):
                for k in range(System_Dim):
                    if h!=k:
                        Diss_time[h][int(i/8)] += Jdis[h][k][h] * intpop[h] + Jdis[h][h][k] * intpop[k]

    Diss_time_avera += Diss_time
    
    
    
Diss_Tavg /= float(traj)

for i in range(System_Dim):
    popavg[i] /= float(traj)
    Diss_avg[i] /= float(traj)
    Diss_time_avg[i] /= float(traj)

for i in range(System_Dim):
    Diss_time_avera[i] /= float(traj)
    
np.savetxt('pop.txt', np.c_[time_axis*TUNIT_TO_PS,popavg[0],popavg[1],popavg[2],popavg[3],popavg[4],popavg[5],popavg[6]],fmt='%.6f')
np.savetxt('D_Total.txt', np.c_[w_axis/WAVENO_TO_EUNIT,Diss_Tavg],fmt='%.6f')
np.savetxt('D_site.txt', np.c_[w_axis/WAVENO_TO_EUNIT,Diss_avg[0],Diss_avg[1],Diss_avg[2],Diss_avg[3],Diss_avg[4],Diss_avg[5],Diss_avg[6]],fmt='%.6f')
np.savetxt('D_Time.txt', np.c_[time_axis*TUNIT_TO_PS,Diss_time_avg[0],Diss_time_avg[1],Diss_time_avg[2],Diss_time_avg[3],Diss_time_avg[4],Diss_time_avg[5],Diss_time_avg[6]],fmt='%.6f')

dissipative = open('Dissipative.txt', 'a')
for i in range(int(ntgrid/24)):
    for zx in range(nmode):
        dissipative.writelines(str(w_axis[zx]/WAVENO_TO_EUNIT)+'\t' +str(Diss_time_avera[0][int(i)][zx])+'\t' +str(Diss_time_avera[1][int(i)][zx])+'\t' +str(Diss_time_avera[2][int(i)][zx])+ '\t' +str(Diss_time_avera[3][int(i)][zx])+'\t' +str(Diss_time_avera[4][int(i)][zx])+'\t' +str(Diss_time_avera[5][int(i)][zx])+'\t' +str(Diss_time_avera[6][int(i)][zx])+'\n')
    dissipative.writelines('\n')
    dissipative.writelines('\n')
dissipative.close()