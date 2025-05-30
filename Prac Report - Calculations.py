import matplotlib.pyplot as plt
import numpy as np
import uncertainties
import uncertainties.unumpy as unp
from uncertainties import ufloat
import uncertainties.umath as umath
from PIL import Image


n,m = 4,5

# Physical Constants
g = 9.81

# Given Constants
pipe_length = 1.22
diameter_mm = np.array([10.92, 13.84, 19.94, 26.0])
pipe_roughness_m = 1.5*(10**-6)*np.ones((n))

# Measures
temperature_C = unp.uarray( [[27.6, 27.7, 27.7, 27.7, 27.7],
                             [27.6, 27.7, 27.8, 27.9, 27.9],
                             [27.9, 27.9, 27.9, 28.0, 28.0],
                             [28.0, 28.0, 28.0, 28.1, 28.1]], 
                            np.ones((n,m))*0.05)
flow_rate_lmin = unp.uarray([[8, 16, 24, 32, 34],
                             [8, 16, 24, 32, 40],
                             [8, 16, 24, 32, 40],
                             [8, 16, 24, 32, 40]],
                            np.ones((n,m))*2/np.sqrt(12))
exp_headloss_mmHg = unp.uarray( [[20.1,   66,     130,    226,    252],
                                 [8.1,    25.2,   48.3,   85,     120.6],
                                 [np.nan,      3.2,     9,      14.5,   20.3],
                                 [np.nan,      np.nan,      np.nan,      3.6,    4.6]],
                                np.ones((n,m))*0.005*1551.8)

## Formulas
def get_density(T):
    return 998.203/(1+0.000207*(T-20))
def get_dynamic_viscosity(T):
    A, B, C = 2.414*10**-5, 247.8, 140
    return A*10**(B/(T+273.15-C))


## Initial Values
diameter_m = diameter_mm*0.001
flow_rate_m3s = flow_rate_lmin*(10**-3)/60
cross_section_area_m2 = np.zeros((n))
velocity_ms = unp.uarray(np.zeros((n,m)), np.zeros((n,m)))
density_kgm3 = unp.uarray(np.zeros((n,m)), np.zeros((n,m)))
dynamic_viscosity_Pas = unp.uarray(np.zeros((n,m)), np.zeros((n,m)))
Re = unp.uarray(np.zeros((n,m)), np.zeros((n,m)))
for i in range(n):
    for j in range(m):
        cross_section_area_m2[i] = np.pi*(diameter_m[i]/2)**2
        velocity_ms[i,j] = flow_rate_m3s[i,j]/cross_section_area_m2[i]
        density_kgm3[i,j] = uncertainties.wrap(get_density)(temperature_C[i,j])
        dynamic_viscosity_Pas[i,j] = uncertainties.wrap(get_dynamic_viscosity)(temperature_C[i,j])
        Re[i,j] = density_kgm3[i,j] * velocity_ms[i,j] * diameter_m[i] / dynamic_viscosity_Pas[i,j]

## Experimental
exp_headloss_Pa = exp_headloss_mmHg * 133.322

## Holland friction factor
haaland_friction_factor = unp.uarray(np.zeros((n,m)), np.zeros((n,m)))
haaland_headloss_m = unp.uarray(np.zeros((n,m)), np.zeros((n,m)))
haaland_headloss_Pa = unp.uarray(np.zeros((n,m)), np.zeros((n,m)))
for i in range(n):
    for j in range(m):
        haaland_friction_factor[i,j] = 1/((-1.8*unp.log(6.9/Re[i,j] + (pipe_roughness_m[i]/diameter_m[i]/3.7)**1.11 , 10) )**2)
        haaland_headloss_m[i,j] = haaland_friction_factor[i,j]*pipe_length*(velocity_ms[i,j]**2)/(2*g*diameter_m[i])
        haaland_headloss_Pa[i,j] = density_kgm3[i,j]*g*haaland_headloss_m[i,j]


# ## Moody approach
relative_pipe_roughness = np.zeros(n)
for i in range(n):
    relative_pipe_roughness[i] = pipe_roughness_m[i]/diameter_m[i]

# Compute from points taken on the image
moody_friction_factor_pixels = np.array([[2997, 3296, 3468, 3532, 3544],
                                         [2920, 3152, 3320, 3444, 3512],
                                         [2736, 3048, 3212, 3332, 3412],
                                         [2564, 2964, 3124, 3248, 3320]])
a_x = (np.log(10**7)-np.log(10**3))/(7053-1335)
b_x = np.log(10**3)-a_x*1335
a_y = (np.log(0.01)-np.log(0.1))/(4646-781)
b_y = np.log(0.01)-a_y*4646
moody_friction_factor = np.exp(a_y*moody_friction_factor_pixels+b_y)

""" fig = plt.figure()
implot = plt.imshow(Image.open('./Moody_Diagram.png'))
colors = ['r', 'g', 'b', 'y']
for i in range(n):
    for j in range(m):
        plt.plot((np.log(Re[i,j].nominal_value)-b_x)/a_x, moody_friction_factor_pixels[i,j], colors[i]+'+')
        plt.annotate(str(i)+':'+str(j), ( (np.log(Re[i,j].nominal_value)-b_x)/a_x, moody_friction_factor_pixels[i,j] ), color=colors[i], textcoords='offset points', xytext=(0,0))    
plt.show() """

moody_headloss_m = unp.uarray(np.zeros((n,m)), np.zeros((n,m)))
moody_headloss_Pa = unp.uarray(np.zeros((n,m)), np.zeros((n,m)))
for i in range(n):
    for j in range(m):
        moody_headloss_m[i,j] = moody_friction_factor[i,j]*pipe_length*velocity_ms[i,j]**2/(2*g*diameter_m[i])
        moody_headloss_Pa[i,j] = g*moody_headloss_m[i,j]*density_kgm3[i,j]


## Errors
haaland_error = np.zeros((n,m))
moody_error = np.zeros((n,m))
theoretical_error = np.zeros((n,m))
for i in range(n):
    for j in range(m):
        haaland_error[i,j] = np.abs(haaland_headloss_Pa[i,j].nominal_value - exp_headloss_Pa[i,j].nominal_value)/exp_headloss_Pa[i,j].nominal_value
        moody_error[i,j] = np.abs(moody_headloss_Pa[i,j].nominal_value - exp_headloss_Pa[i,j].nominal_value)/exp_headloss_Pa[i,j].nominal_value
        theoretical_error[i,j] = 2*np.abs(haaland_headloss_Pa[i,j].nominal_value - moody_headloss_Pa[i,j].nominal_value)/(haaland_headloss_Pa[i,j].nominal_value + moody_headloss_Pa[i,j].nominal_value)

## Plotting
fig, ax = plt.subplots()
i=0
plt.errorbar(unp.nominal_values(velocity_ms[i]), unp.nominal_values(exp_headloss_Pa[i]), xerr=unp.std_devs(velocity_ms[i]), yerr=unp.std_devs(exp_headloss_Pa[i]), ecolor='r', fmt='none', elinewidth=0.5, capsize=4, capthick=0.5)
#plt.errorbar(unp.nominal_values(velocity_ms[i]), unp.nominal_values(moody_headloss_Pa[i]), xerr=unp.std_devs(velocity_ms[i]), yerr=unp.std_devs(moody_headloss_Pa[i]), ecolor='b', fmt='none', elinewidth=0.5, capsize=4, capthick=0.5)
#plt.errorbar(unp.nominal_values(velocity_ms[i]), unp.nominal_values(haaland_headloss_Pa[i]), xerr=unp.std_devs(velocity_ms[i]), yerr=unp.std_devs(haaland_headloss_Pa[i]), ecolor='g', fmt='none', elinewidth=0.5, capsize=4, capthick=0.5)
plt.fill_between(unp.nominal_values(velocity_ms[i]), 
                 unp.nominal_values(moody_headloss_Pa[i]) + unp.std_devs(moody_headloss_Pa[i]),
                 unp.nominal_values(moody_headloss_Pa[i]) - unp.std_devs(moody_headloss_Pa[i]),
                 color = 'b',
                 alpha = 0.4,
                 edgecolor = 'none',
                 label='Moody headloss uncertainty')
plt.fill_between(unp.nominal_values(velocity_ms[i]), 
                 unp.nominal_values(haaland_headloss_Pa[i]) + unp.std_devs(haaland_headloss_Pa[i]),
                 unp.nominal_values(haaland_headloss_Pa[i]) - unp.std_devs(haaland_headloss_Pa[i]),
                 color = 'g',
                 alpha = 0.4,
                 edgecolor = 'none',
                 label='Haaland headloss uncertainty')
plt.plot(unp.nominal_values(velocity_ms[i]), unp.nominal_values(exp_headloss_Pa[i]), 'r+', label='Experimental Headloss')
plt.plot(unp.nominal_values(velocity_ms[i]), unp.nominal_values(haaland_headloss_Pa[i]), 'g+', label='Haaland Headloss')
plt.plot(unp.nominal_values(velocity_ms[i]), unp.nominal_values(moody_headloss_Pa[i]), 'b+', label='Moody Headloss')
plt.title('Headloss (Pa) in a straight pipe, l='+str(pipe_length)+'m, d='+str(diameter_mm[i])+'mm')
plt.legend()
plt.xlabel('Velocity (m/s)')
plt.ylabel('Headloss (Pa)')
plt.show()
