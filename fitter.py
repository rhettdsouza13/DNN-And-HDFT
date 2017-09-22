from scipy.optimize import curve_fit
import numpy

def func(x,a,b):
    return (1-(numpy.exp(-a*(x))) - b)

def fitter(x_in,y):
    popt, pcov = curve_fit(func, x_in, y, p0 = [2.5, -0.03])

    return popt, pcov
