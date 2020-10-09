import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from atlas_fit_function import atlas_invMass_mumu_core

# Load data: 
inFileName = "DATA/original_histograms/mass_mm_higgs_Background.npz"
with np.load(inFileName) as data:
	bin_edges = data['bin_edges']
	bin_centers = data['bin_centers']
	bin_values = data['bin_values']
	bin_errors = data['bin_errors']
 
#############################################################################################################################	
# 1. Fit poly3

def poly3(x, a, b, c, d):
	return a*x**3 + b*x**2 + c*x +d

fitfun = poly3
popt, pcov = curve_fit(fitfun,bin_centers,bin_values,sigma=bin_errors, p0=[1,1,1,1])
perr = np.sqrt(np.diag(pcov))
a,b,c,d = popt

my_fit=np.array(fitfun(bin_centers, a,b,c,d))

xerrs = 0.5*(bin_edges[1:]-bin_edges[:-1])

plt.figure()
plt.errorbar(bin_centers, bin_values, bin_errors, xerrs, fmt="none",color='b',ecolor='b',label='Original histogram')
plt.plot(bin_centers,my_fit,'g-', label='fit poly3')
plt.legend()