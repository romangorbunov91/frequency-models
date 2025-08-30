import numpy as np

def reactor_model(freq, L, r, C):
   
    s = 1j * 2 * np.pi * freq
    
    # Parameters.
    L *= 136e-6
    r *= 0.025
    C *= 725e-12
    #R *= 6e3
    
    # Complex gain.
    #gain_comp = (1 /(s*C) * (r + s*L) / (1 /(s*C) + (r + s*L)) * R) / (1 /(s*C) * (r + s*L) / (1 /(s*C) + (r + s*L)) + R)
    #gain_comp = 1 /(s*C) * (r + s*L) / (1 /(s*C) + (r + s*L))
    gain_comp = (r + s*L) / (1 + (s*C)*(r + s*L))
    #gain_comp = (r + s*L) * C
    #gain_comp = R /(s*C) / (R + 1/(s*C)) * (r + s*L) / (R /(s*C) / (R + 1/(s*C)) + (r + s*L))
    
    # Magnitude.
    gain_abs = 20*np.log10(np.abs(gain_comp))
    #gain_abs = np.abs(gain_comp)
    gain_phase = np.unwrap(np.angle(gain_comp))*180/np.pi

    return [gain_abs, gain_phase]

def reactor_model_gain_abs(freq, L, r, C):
   
    w = 2 * np.pi * freq
    
    # Parameters.
    L *= 136e-6
    r *= 0.025
    C *= 725e-12
    #R *= 6e3
    
    gain = np.sqrt((r**2 + (w*L)**2) / ((1 - w**2*L*C)**2 + (w*r*C)**2))
    
    # Magnitude.
    gain_db = 20*np.log10(np.abs(gain))

    return gain_db


def grad_reactor_model_gain_abs(freq, L, r, C):
    w = 2 * np.pi * freq
    
    # Parameters.
    L *= 136e-6
    r *= 0.025
    C *= 725e-12
    #R *= 6e3
    
    grad = np.zeros((3,))
    grad[0] = w**2*L*(1 - C*L*w**2 + C/L*r**2)
    grad[1] = r*(1 - 2*C*L*w**2)
    grad[2] = w**2*L*(r**2 + (w*L)**2) * (1 - C*L*w**2 - C/L*r**2)
    grad *= 1 / np.sqrt((r**2 + (w*L)**2)/((1 - w**2*L*C)**2 + (w*r*C)**2)) / ((1 - w**2*L*C)**2 + (w*r*C)**2)**2
    
    return grad

def grad_func(freq, y, w):

    gain_db_dataset = reactor_model_gain_abs(freq, L=w[0], r=w[1], C=w[2])#, R)
    grad_dataset = grad_reactor_model_gain_abs(freq, L=w[0], r=w[1], C=w[2])#, R)
    grad = np.zeros((3,))
    for k in range(len(grad)):
        grad[k] = np.sum(2*(gain_db_dataset - y)*grad_dataset[k])
    return grad


# freq - частоты.
# y - эталон отклика.
def loss_func(freq, y, w):

    [abs_dataset, _] = reactor_model(freq, L=w[0], r=w[1], C=w[2])#, R)
    #[_, phase_dataset] = reactor_model(freq, L=w[0], r=w[1], C=w[2])#, R)

    # Функция возвращает сумму квадратов отклонений наблюдений от эталона.
    return np.sum((abs_dataset - y)**2)