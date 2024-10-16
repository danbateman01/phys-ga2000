
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

hdu_list = fits.open('specgrid.fits')
logwave = hdu_list['LOGWAVE'].data
flux = hdu_list['FLUX'].data

fn = 0
def fignum():
    global fn
    fn +=1
    return fn

#A
for i in range(0,5):
    plt.plot(logwave, flux[i, :], label="Galaxy {galaxyindex}".format(galaxyindex = i))
plt.ylabel(r'Flux [$10^{-17}$ \mathrm{erg} s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]')
plt.xlabel('Wavelength [$A$]')
plt.legend()
plt.title('Part A: Spectrum of the First Galaxy')
plt.savefig("ps6a.png")
plt.show()

#B

flux_sum = np.sum(flux, axis=1)

flux_normalized = flux / np.tile(flux_sum, (np.shape(flux)[1], 1)).T

# C
means_normalized = np.mean(flux_normalized, axis=1)
flux_normalized_0_mean = flux_normalized - np.tile(means_normalized, (np.shape(flux)[1], 1)).T


#D
def sorted_eigs(r, return_eigvalues = False):

    corr=r.T@r
    eigs=np.linalg.eig(corr) 
    arg=np.argsort(eigs[0])[::-1] 
    eigvec=eigs[1][:,arg] 
    eig = eigs[0][arg]
    if return_eigvalues == True:
        return eig, eigvec
    else:
        return eigvec

r = flux_normalized_0_mean 
r_subset = r[:500, :]
logwave_subset = logwave
C = r_subset.T@r_subset 
C.shape
r_subset.shape
eigvals, eigvecs = sorted_eigs(r_subset, return_eigvalues = True)

for i in range(5):
    plt.plot(logwave_subset, eigvecs[:, i], label=f'Eigenvector {i+1}')
plt.legend()
plt.ylabel('Normalized 0-mean Flux')
plt.xlabel('Wavelength [$A$]')
plt.title('Part D: Normalised vs Wavelength')
#plt.savefig("ps6d.png")
plt.show()

# Part E

U, S, Vh = np.linalg.svd(r_subset, full_matrices=True)
eigvecs_svd = Vh.T
eigvals_svd = S**2
svd_sort = np.argsort(eigvals_svd)[::-1]
eigvecs_svd = eigvecs_svd[:,svd_sort]
eigvals_svd = eigvals_svd[svd_sort]

[plt.plot(eigvecs_svd[:,i], eigvecs[:,i], 'o')for i in range(500)]
plt.plot(np.linspace(-0.2, 0.2), np.linspace(-0.2, 0.2))
plt.xlabel('SVD eigenvalues')
plt.ylabel('Eig eigenvalues')
plt.title("Comparing SVD and Eig eigenvalues")
plt.show()

plt.plot(eigvals_svd, eigvals[:500], 'o')
plt.xlabel('SVD eigenvalues')
plt.ylabel('Eig eigenvalues')
plt.title("SVD ")
#plt.savefig("ps6e")
plt.show()

#G

def PCA(l, r, project = True):

    eigvector = sorted_eigs(r)
    eigvec=eigvector[:,:l] 
    reduced_wavelength_data= np.dot(eigvec.T,r.T) 
    if project == False:
        return reduced_wavelength_data.T 
    else: 
        return np.dot(eigvec, reduced_wavelength_data).T

nc = 5
plt.plot(logwave_subset, PCA(nc,r_subset)[1,:], label = 'l = {Nc}'.format(Nc = nc))
plt.plot(logwave_subset, r_subset[1,:], label = 'original data')

plt.ylabel('normalized 0-mean flux')
plt.xlabel('wavelength [$A$]')
plt.title('Result of PCA Nc = {Nc}')
plt.legend()
#plt.savefig("ps6g.png")
plt.show()

#h

c0 = eigvecs[:, 0]  # Coefficient 0
c1 = eigvecs[:, 1]  # Coefficient 1
c2 = eigvecs[:, 2]  # Coefficient 2

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(c0, c1, s=10, alpha=0.5)
plt.xlabel('Coefficient 0')
plt.ylabel('Coefficient 1')
plt.title('c0 vs c1')

plt.subplot(1, 2, 2)
plt.scatter(c0, c2, s=10, alpha=0.5)
plt.xlabel('Coefficient 0')
plt.ylabel('Coefficient 2')
plt.title('c0 vs c2')

plt.tight_layout()
#plt.savefig("ps6h.png")
plt.show()

#I

residuals = []
Nc_values = np.arange(1,21)

for Nc in Nc_values:
   
    approximated_spectra = PCA(Nc, r_subset, project=True)
    squared_difference = np.sum((r_subset - approximated_spectra) ** 2)


    original_squared_sum = np.sum(r_subset ** 2)

    
    residual = squared_difference / original_squared_sum

    residuals.append(residual)


plt.plot(Nc_values,residuals, 'o')
plt.xlabel('Number of Coefficients')
plt.ylabel(r'Flux [$10^{-17}$ \mathrm{erg} s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]^2')
plt.title('Squared Fractional Residuals vs. Nc')
plt.grid(True)
plt.show()
print('Squared Fractional Error for Nc = 20: {squared_residuals[-1]:.6f}')