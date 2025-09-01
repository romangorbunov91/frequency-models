# version 1.0 by romangorbunov91
# 01-Sep-2025

import numpy as np
from enum import IntEnum

class var(IntEnum):
    freq, mag_db, ph_deg = range(3)

def reactor_model_gain(freq, L, r, C):
   
    s = 1j * 2 * np.pi * freq
    
    # Complex gain.
    #gain_comp = (1 /(s*C) * (r + s*L) / (1 /(s*C) + (r + s*L)) * R) / (1 /(s*C) * (r + s*L) / (1 /(s*C) + (r + s*L)) + R)
    #gain_comp = 1 /(s*C) * (r + s*L) / (1 /(s*C) + (r + s*L))
    gain_comp = (r + s*L) / (1 + (s*C)*(r + s*L))
    #gain_comp = (r + s*L) * C
    #gain_comp = R /(s*C) / (R + 1/(s*C)) * (r + s*L) / (R /(s*C) / (R + 1/(s*C)) + (r + s*L))
    
    # Magnitude.
    gain_abs = 20*np.log10(np.abs(gain_comp))
    gain_phase = np.unwrap(np.angle(gain_comp))*180/np.pi

    return [gain_abs, gain_phase]

def reactor_model_gain_abs(freq, L, r, C):
   
    omega = 2 * np.pi * freq

    return 20*np.log10(np.sqrt((r**2 + (omega*L)**2) / ((1 - L*C*omega**2)**2 + (omega*r*C)**2)))

def grad_func(freq, y, w):
    # Base parameters.
    L_b = 100e-6
    r_b = 10e-3
    C_b = 100e-12
    
    # Parameters.
    L = w[0] * L_b
    r = w[1] * r_b
    C = w[2] * C_b
    #R *= 6e3
    
    gain_db_dataset = reactor_model_gain_abs(freq, L, r, C)#, R)
    
    omega = 2 * np.pi * freq   
    
    grad = np.zeros((len(w),))
    
    for k in range(len(grad)):
        if k == 0:
            grad_dataset = L*omega**2 / (r**2 + (omega*L)**2) * (1 - C*L*omega**2 + C/L*r**2)
            # Additional scaling because of derivative by L, not by w[0].
            grad_dataset *= L_b
        elif k == 1:
            grad_dataset = r / (r**2 + (omega*L)**2) * (1 - 2*C*L*omega**2)
            # Additional scaling because of derivative by r, not by w[1].
            grad_dataset *= r_b
        elif k == 2:
            grad_dataset = L*omega**2*(1 - C*L*omega**2 - C/L*r**2)
            # Additional scaling because of derivative by C, not by w[2].
            grad_dataset *= C_b
        
        grad_dataset *= 20/np.log(10)/((1 - L*C*omega**2)**2 + (omega*r*C)**2)
        grad[k] = np.sum(2*(gain_db_dataset - y)*grad_dataset)

    return grad/len(freq)

def loss_func(freq, y, w):
    # Base parameters.
    L_b = 100e-6
    r_b = 10e-3
    C_b = 100e-12
    
    # Parameters.
    L = w[0] * L_b
    r = w[1] * r_b
    C = w[2] * C_b
    #R *= 6e3

    gain_db_dataset = reactor_model_gain_abs(freq, L, r, C)#, R)    

    # Функция возвращает сумму квадратов отклонений наблюдений от эталона.
    return np.sum((gain_db_dataset - y)**2)/len(freq)

'''
def grad_func(freq, y, w):

    gain_db_dataset = reactor_model_gain_abs(freq, L=w[0], r=w[1], C=w[2])#, R)
    omega = 2 * np.pi * freq
    
    # Parameters.
    L = w[0] * 136e-6
    r = w[1] * 0.025
    C = w[2] * 725e-12
    #R *= 6e3
    
    grad = np.zeros((len(w),))
    
    for k in range(len(grad)):
        if k == 0:
            grad_dataset = omega**2*L*(1 - C*L*omega**2 + C/L*r**2)
        elif k == 1:
            grad_dataset = r*(1 - 2*C*L*omega**2)
        elif k == 2:
            grad_dataset = omega**2*L*(r**2 + (omega*L)**2)*(1 - C*L*omega**2 - C/L*r**2)
        
        grad_dataset *= 1/np.sqrt((r**2 + (omega*L)**2)/((1 - omega**2*L*C)**2 + (omega*r*C)**2))/((1 - omega**2*L*C)**2 + (omega*r*C)**2)**2
            
        grad[k] = np.sum(2*(gain_db_dataset - y)*grad_dataset)
        
    return grad
'''