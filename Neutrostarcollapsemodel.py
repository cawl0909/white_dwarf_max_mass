# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 17:29:28 2024

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

THIS VERSION IS FOR WHITE DWARVES

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


@author: awwab
"""

import numpy as np
import scipy.constants as consts
import astropy.constants as astroconsts
import math
import matplotlib.pyplot as plt

#CONSTANTS ALL UNITS ARE IN THE CGS SCALE1!!!!!!

gravconst = 6.6732*(10**-8)

white_dwarf_centeral_pressure = 2.3302*(10**22)# dyne^2/cm^3 for white dwarf
white_dwarf_centeral_pressure_non_rel = 5.62*(10**25)

neutron_star_centeral_pressure =  5.0*(10**32)

solar_mass = astroconsts.M_sun.value*1000
r0 =  100*consts.G*(astroconsts.M_sun.value)/((consts.c**2))# 1/2 Schwartzchild radius
hbar = consts.hbar*(10**7)
e_mass =  consts.m_e*1000 #g
n_mass = consts.m_n*1000 #g
p_mass = consts.m_p*1000 #g
nucleon_mass = n_mass   #(n_mass)
c =  consts.speed_of_light*100 #cms^-1

epsilon_0 =  (n_mass**4)*(c**5)/((np.pi**2)*(hbar**3))

####
#ALL DONE IN CM/GRAMS 

##
#K formulae

knonrel = ((hbar**2)/(15*(consts.pi**2)*e_mass))*(((3*(consts.pi**2))/(2*(nucleon_mass)*(c**2)))**(5/3))
knonrel_n = ((hbar**2)/(15*(consts.pi**2)*n_mass))*(((3*(consts.pi**2))/(1*(nucleon_mass)*(c**2)))**(5/3))

krel = ((hbar*c)/(12*(consts.pi**2)))*((((3/2)*(consts.pi**2))/(nucleon_mass*(c**2)))**(4/3))
krel_n = ((hbar*c)/(12*(consts.pi**2)))*((((3)*(consts.pi**2))/(nucleon_mass*(c**2)))**(4/3))

#ODE equations

#equation of state is [pressure, mass]
gamma = 3/5 #3/5 and 3/4 for non and relatavistic cases
#pressure ode #pa/m -> (dyne/cm2)/km

### POLYTROPIC APPROXS####################################

def dpdr(r,state):
    dpdr_variable = (-1)*((state[0])**(gamma))*(state[1])/(r**2)
    dpdr_coeff = (r0)/((knonrel_n)**(gamma))
    t_dpdr = dpdr_variable*dpdr_coeff
    #print(t_dpdr,r,state[0],state[1])
    return t_dpdr

#mass ode, units of stellar mass should be in (m/sm)/cm
def dmdr(r,state):
    dmdr_coeff = 4*(consts.pi)/(c**2)/((knonrel_n)**(gamma))
    dmdr_variable = (r**2)*((state[0])**(gamma))
    t_dmdr =  dmdr_variable*dmdr_coeff/solar_mass
    return t_dmdr

#########################################################
#Newtonian ODES

def dpdr_newtonian(r,state):
    epsilon = epsilon_poly_approx(r, state)
    tt1 = -gravconst*(epsilon)*state[1]/(c**2)/(r**2)
    return tt1*solar_mass

def dmdr_newtonian(r,state):
    epsilon = epsilon_poly_approx(r, state)
    tt1 = (4*consts.pi)*(r**2)*epsilon/(c**2)
    return tt1/solar_mass


#########################################################
#print(dmdr(1,state_start))
def dpdr_tov(r,state):
    epsilon = epsilon_poly_approx(r,state)
    mass = state[1]*solar_mass
    tt1 = -gravconst*(epsilon + state[0])*((c**2)*mass+4*consts.pi*(r**3)*state[0])
    tt2 =  (c**2)*(r)*(((c**2)*(r))-2*gravconst*mass)
    return (tt1/tt2)

def epsilon_full(r):
    return

def epsilon_poly_approx(r,state):
    x =  (15*state[0]/epsilon_0)**(1/5)
    return ((1/3)*epsilon_0)*(x**3)

def pressure_tov_full(x,p0):
    temp_coeff = (epsilon_0)/24
    tterm1 = (2*x**3 -3*x)*((1+x**2)**(1/2)) + 3*(math.asinh(x))
    return (temp_coeff*tterm1)-p0

def pressure_tov_full_dx(x):
    temp_coeff = (epsilon_0)/24
    tterm1 = (8*x**4-6)
    tterm2 = ((x**2+1)**(1/2))
    return temp_coeff*tterm1/tterm2

###############################################################################

#NEWTON RAPHSON ROOT FINDING

###############################################################################

zero_tolerance = 0.01
start_val = 1
max_iter = 1000

def newton_rootfinder(func,funcdr,tolerance,maxiter,start,pressure0):
    old_x = start
    for i in range(0,max_iter):
        new_x = old_x - func(old_x,pressure0)/funcdr(old_x)
        old_x = new_x
        if abs(func(old_x,pressure0)) <= zero_tolerance:
            break
        elif (i == max_iter-1):
            #print("timeout")
            continue
    return old_x

def tesfuncddx(x):
    return (epsilon_0/5)*(x**4)
def testfunc(x,p0):
    return ((epsilon_0/15)*(x**5)-p0)
def testfunc1(x,p0):
    return(x-3)

def newton_rootfinder_num(func, tolerance, maxiter, start, pressure0):
    old_x = start
    for i in range(0,max_iter):
        funcdr =  numerical_deriv(func, old_x, 0.01,pressure0)
        new_x = old_x - func(old_x,pressure0)/funcdr
        old_x = new_x
        if abs(func(old_x,pressure0)) <= zero_tolerance:
            break
        elif (i == max_iter-1):
            #print("timeout")
            continue
    return old_x
    return
###############################################################################

#Numerical derivative

###############################################################################

def numerical_deriv(func,x,dx,p0):
    deriv = (func(x+dx,p0)-func(x-dx,p0))/dx
    return deriv

###############################################################################

#RK4 ODE SOLVER

###############################################################################

m,k = 3,5
def grad(r,state):
    x =  state[1]
    v = -(k/m)*state[0]
    return np.array([x,v])

def total_ode(r,statein):
    state = statein.copy()
    pressure =  dpdr(r,state)
    mass = dmdr(r,state)
    #print(pressure,mass)
    return np.array((pressure,mass))

def total_ode_tov(r,statein):
    state2 = statein.copy()
    pressure = dpdr_tov(r, state2)
    mass  = dmdr_newtonian(r,state2)
    return np.array((pressure,mass))

def total_ode_newtonian(r,statein):
    state = statein.copy()
    pressure = dpdr_newtonian(r, state)
    mass = dmdr_newtonian(r, state)
    return np.array((pressure,mass))
    

def rk4_step(func,state,r,dr):
    if (r==0):
        k1 = func(r+0.000000000001,state)*dr
    else:
        k1 = func(r,state)*dr
    k2 = func(r+dr/2,state+k1/2)*dr
    k3 = func(r+dr/2,state+k2/2)*dr
    k4 = func(r+dr,state+k3)*dr
    final_state = state + (1/3)*(k1/2 + k2 +k3 + k4/2)
    return final_state

def rk4_full(centeral_pressure,ode):
    states_array = np.zeros((0,2))
    r_start = 0
    r_max = 5000000.0
    dr_step =  1000.0
    r_array = np.arange(r_start,r_max+ dr_step,dr_step)
    init_state = np.array([centeral_pressure,0])
    states_array =  np.vstack((states_array,init_state))
    for r in range(1,len(r_array)):
        #print(r)
        temp_state = rk4_step(ode, states_array[-1], r_array[r-1], dr_step)
        if (np.isnan((temp_state)).any() == True):
            break
        states_array = np.vstack((states_array,temp_state))
    r_array = r_array[:len(states_array)]
    #print(len(states_array),len(r_array))
    return (np.c_[r_array,states_array]) ##adds column

def rk4_return_max(centeral_pressure,ode):
    temp = rk4_full(centeral_pressure,ode)
    return(np.array([temp[-1,0],temp[-1,2]]))

test = rk4_full(neutron_star_centeral_pressure,total_ode)

#pressure_list = np.arange(2*10**32,(2*10**22),(1*10**32),dtype="float")
#pressure_list = np.linspace(0.0, 4.5*10**22,30)
pressure_list = np.logspace(np.log10(2.0*10**31), np.log10(4.0*10**33),100)
values_list = np.zeros((0,3))
values_list2 = np.zeros((0,3))

for i in range(0,len(pressure_list)):
    print(pressure_list[i])
    temp_out = rk4_return_max(pressure_list[i],total_ode)
    values_list = np.vstack((values_list,[pressure_list[i],temp_out[0],temp_out[1]]))
    temp_out2 = rk4_return_max(pressure_list[i], total_ode_tov)
    values_list2 = np.vstack((values_list2,[pressure_list[i],temp_out2[0],temp_out2[1]]))



##  PLOTTING ########################

fig = plt.figure(figsize  = (12,10))
ax1 = fig.add_subplot(211)
ax2 = ax1.twinx()
#ax3 = fig.add_subplot(212)
#ax4 = ax3.twinx()
#ax1.set_xlabel("r (cm)")
ax1.set_xlabel("p0 ($dyne/cm^2$)")
#ax1.set_ylim((0.5*(10**9)),(1.7*(10**9)))
#ax2.set_ylim(0,1.6)
ax1.set_title('Neutron Star Non-Relatavistic Polytropc approxiamtion and TOV')

#ax1.set_ylabel('Pressure dyne/cm^2',color="red")
#ax2.set_ylabel('M in SM',color="blue")

ax1.set_ylabel('Radius ($cm$)',color="red")
ax2.set_ylabel('Mass SM ratio ($M/M_{\odot}$)',color="blue")

#ax1.plot(test[:,0],test[:,1],label = "Pressure",color="red")
#ax2.plot(test[:,0],test[:,2],label = "% Mass")

ax1.plot(values_list[:,0],values_list[:,1],label = "Radius Poly",color="red")
ax1.plot(values_list2[:,0],values_list2[:,1],label = "Radius TOV",color="red",linestyle="dashed")

ax2.plot(values_list[:,0],values_list[:,2],label = "Mass Poly",color="blue")
ax2.plot(values_list2[:,0],values_list2[:,2],label = "Mass TOV",color="blue",linestyle="dashed")

test2 = rk4_full(neutron_star_centeral_pressure,total_ode_tov)

#ax1.plot(test2[:,0],test2[:,1],label = "Radius",color="red")
#ax2.plot(test2[:,0],test2[:,2],label = "Radius",color="blue")

#ax3.plot(test[:,0],test[:,1],color="orange")

#ax4.plot(test[:,0],test[:,2],color="purple")
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc="upper center")

ax1.set_xscale('log')
ax1.margins(x=0)
plt.savefig("test1",dpi=200,bbox_inches="tight")
plt.show()
plt.clf()
# DATA OUTPUT

#print(test[:,1:3])
print("------test1-----------")
print((test))
print("-----test2------------")
print((test2))

print("------values-----------")
print(values_list)
print("------values2-----------")
print(values_list2)
# Show and close the plot
print("------consts-----------")
print(knonrel_n)
print(epsilon_0)
print(gravconst)
print("-----TESTS------------")

#testx= ((15*(neutron_star_centeral_pressure)/epsilon_0)**(1/5))
#testep = (epsilon_0*(1/3)*(testx**3))
#print(testx)
#test_newton = (newton_rootfinder(testfunc,tesfuncddx, 0.01, 1000, 1,neutron_star_centeral_pressure))
#print(test_newton)
#test_newton = (newton_rootfinder_num(testfunc, 0.01, 1000, 1,neutron_star_centeral_pressure))
#print(test_newton)
#test_newton = (newton_rootfinder_num(pressure_tov_full, 0.01, 1, 1,neutron_star_centeral_pressure))
#print(test_newton)
#test_newton = (newton_rootfinder_num(testfunc1, 0.1, 10000, 4,neutron_star_centeral_pressure))
#print(test_newton)