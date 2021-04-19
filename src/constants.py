from math import pi, sin, cos
import numpy as np

#list of elementary constants
u0 = 12.56637e-7
e = 1.6e-19                   #Elementary charge in C
h_bar = 1.054e-34             #Reduced Planck constant in Js
uB = 9.274e-24                #Bohr magneton in J/T
gamma = 2*u0*uB/h_bar         #Gyromagnetic ratio in m/As
kb = 1.38e-23                 #Boltzmann constant
m = 9.11e-31                  #Electron Mass


#list of physical parameters
# # Electronic Module, Dynamic switching
l = 60e-9                     # Length in m
w = 50e-9                     # Width in m
d = 3e-9                      # Thickness in m
A2 = d*w                      # Cross-sectional area (the rectangular)
rho = 278e-8                  # Resistivity of beta-IrMn
R_W = rho*l/(w*d)             # Resistance of beta-IrMn

# # Anisotropy Module, Initialization, Dynamic switching
Ki = 0.32e-3                  # The anisotropy energy in J/m2
Ms = 0.625e6                  # Saturation magnetization A/m
beta = 60e-15                 # VCMA coefficient in J/vm, =75 fJ/vm 
tf = 1.1e-9                   # Free layer thickness
tox =1.4e-9                   # Barrier thickness
D = 50e-9                     # Diameter of MTJ
A1 = pi*(D**2)/4              # MTJ Area (the cylinder)
v = tf*A1                     # The volumn of free layer 
T = 300                       # Temperature in K
alpha = 0.05                  # Gilbert Damping Coefficent 
t_step = 1e-12                # Time step of magnetization revolution
Heff = (2*Ki)/(tf*Ms*u0)

# # TMR module
Vh = 0.5                      # Bias voltage at which TMR is divided by 2 

# # Dynamic Switching module
P = 0.58                      # Spin polarization of the tunnel curdsdrent 0.6
theta_SH = 0.25               # Spin Hall angle         
alpha = 0.05

## Initialize
phi_bar = 0.4                 # Potential barrier height ,V
TMR = 1
RA = 650e-12                  # resistance x area