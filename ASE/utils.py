import numpy as np 
from scipy.special import expit

def exp_(x):
    back = None
    try:
        back = np.exp(x)
    except OverflowError:
        back = expit(x)
    if back.any() == np.infty:
        back = expit(x)
    return back

