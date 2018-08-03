from scipy.optimize import curve_fit
import numpy

def func(x,a,b,c):
    return -(c*numpy.exp(-a*(x))) + b

def fitter(x_in,y):
    popt, pcov = curve_fit(func, x_in, y, p0=[0.5,0.9,0.05])
    fitted_plot = [func(i,popt[0],popt[1],popt[2]) for i in range(len(y))]
    print popt
    return fitted_plot
