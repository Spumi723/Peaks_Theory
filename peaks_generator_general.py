"""Generate spherical peak profiles from an arbitrary isotropic 3D power spectrum.

CSV format (no header): first row is the radial distances (in units of R_star), then each subsequent row 
is a profile along one of the six axes from a peak. Each realization is appended to the same CSV file.

The peak finding algorithm uses a threshold under which a peak is not considered (to avoid saving low peaks)
and a local-maximum filter with periodic boundaries.
To identify a maximum we check that the value at a grid cell is the maximum in a cube of side 2*MIN_SEPARATION+1 centered on that cell.
If the power spectrum at the smallest scales of the grid is enough suppressed, the local maxima will be well separated and this will 
not miss any peaks.

For each peak found we save the profile along the positive and negative x, y, and z axes.
"""

from pathlib import Path
import csv
import numpy as np
from scipy.integrate import quad
from scipy.ndimage import maximum_filter

# ---------------------------- USER SETTINGS -------------------------------
N = 256                       # grid cells per side
L = 40.0                      # physical box length
N_REALIZATIONS = 2500         # number of independent realizations
SEED = None                   # integer for random reproducibility
PEAK_THRESHOLD = 2.4          # peak height threshold to save the profile
MIN_SEPARATION = 2            # local-maximum finder neighbourhood radius, in cells
R_MAX_RSTAR = 5.0             # maximum radius for the saved profile, in units of R_star
NORMALIZATION = "theory"       # "sample" (per realization) or "theory"
filename = "test"        # output CSV file
OUTPUT = Path(filename + ".csv")

BETA_NG = -2.0

# Define the Power spectrum P(k)
def power_spectrum(k):
    A, n, R_f = 1e-1, 0, 1.0
    return A * np.asarray(k)**n * np.exp(-(np.asarray(k) * R_f)**2)

# Integration limits for spectral moments: choose to cover the support of P(k).
K_MIN = 0.0
K_MAX = 30.0
# --------------------------------------------------------------------------


def evaluate_pk(k):
    """Evaluate P(k) consistently for scalar integration and FFT arrays."""
    k = np.asarray(k, dtype=float)
    result = np.asarray(power_spectrum(k), dtype=float)
    result = np.broadcast_to(result, k.shape)
    if np.any(~np.isfinite(result)) or np.any(result < 0):
        raise ValueError("P(k) must be finite and nonnegative on its domain")
    return result


def spectral_parameters(kmin=K_MIN, kmax=K_MAX):
    """Continuum moments for the user-supplied P(k), independent of the realized field."""
    if not 0 <= kmin < kmax:
        raise ValueError("Require 0 <= K_MIN < K_MAX")
    variances = []
    for j in range(3):
        integrand = lambda k: k**(2 + 2*j) * float(evaluate_pk(k)) / (2*np.pi**2)
        val, err = quad(integrand, kmin, kmax, epsrel=1e-8, limit=400)
        if not np.isfinite(val) or val <= 0:
            raise ValueError(f"Invalid spectral moment sigma_{j}^2: {val}")
        variances.append(val)
    sigma0, sigma1, sigma2 = np.sqrt(variances)
    gamma = sigma1**2 / (sigma0*sigma2)
    rstar = np.sqrt(3.0) * sigma1 / sigma2
    if not (0 < gamma <= 1.0 + 1e-8):
        raise ValueError(f"Invalid gamma={gamma}; check P(k) and integration limits")
    return sigma0, sigma1, sigma2, gamma, rstar


def fft_spectrum(n, box_length):
    """P(|k|) on the FFT grid; zero DC and modes outside integration limits."""
    k1 = 2*np.pi*np.fft.fftfreq(n, d=box_length/n)
    kx, ky, kz = np.meshgrid(k1, k1, k1, indexing="ij")
    kmag = np.sqrt(kx*kx + ky*ky + kz*kz)
    pk = np.zeros_like(kmag)
    mask = (kmag > 0) & (kmag >= K_MIN) & (kmag <= K_MAX)
    pk[mask] = evaluate_pk(kmag[mask])
    return pk


def generate_field(rng, sqrt_pk, n, box_length):
    """Real periodic Gaussian field with continuum convention <|delta_k|²>=V P(k)."""
    white = rng.standard_normal((n, n, n))
    # np.fft.fftn(white) has variance N³ per mode. Multiplication by
    # sqrt(P / dx³) produces the continuum power-spectrum convention.
    fourier = np.fft.fftn(white) * sqrt_pk / (box_length/n)**1.5
    fourier[0, 0, 0] = 0
    field = np.fft.ifftn(fourier).real
    if not BETA_NG == 0:
        field = -1/BETA_NG * np.log(1 - BETA_NG * field)
        field -= np.average(field)
    return field


def find_peaks(field, threshold, min_separation):
    """Indices of strict-threshold local maxima, with periodic boundaries."""
    neighbourhood = 2*min_separation + 1
    mask = (field == maximum_filter(field, size=neighbourhood, mode="wrap"))
    mask &= field > threshold
    return np.argwhere(mask)

def x_axis_profile(field, L, center, r_max, rstar, physical=True):
    """
    Extract the field profile along the positive x-axis
    from a peak, using periodic boundary conditions.

    Returns
    -------
    r : ndarray
        Distance from the peak, starting at r=0.
    profile : ndarray
        Field values along the positive x-axis.
    """
    N = field.shape[0]
    dx = L / N

    ix, iy, iz = (int(i) for i in center)

    if r_max > L / 2:
        raise ValueError("r_max must not exceed L/2.")

    n_steps = int(np.floor(r_max / dx))
    offsets = np.arange(n_steps + 1)

    x_indices = (ix + offsets) % N

    r = offsets * dx / rstar
    profile = field[x_indices, iy, iz]
    return r, profile

def x_neg_axis_profile(field, L, center, r_max, rstar, physical=True):
    """
    Extract the field profile along the negative x-axis
    from a peak, using periodic boundary conditions.

    Returns
    -------
    r : ndarray
        Distance from the peak, starting at r=0.
    profile : ndarray
        Field values along the negative x-axis.
    """
    N = field.shape[0]
    dx = L / N

    ix, iy, iz = (int(i) for i in center)

    if r_max > L / 2:
        raise ValueError("r_max must not exceed L/2.")

    n_steps = int(np.floor(r_max / dx))
    offsets = np.arange(n_steps + 1)

    x_indices = (ix - offsets) % N

    r = offsets * dx / rstar
    profile = field[x_indices, iy, iz]
    return r, profile

def y_axis_profile(field, L, center, r_max, rstar, physical=True):
    """
    Extract the field profile along the positive y-axis
    from a peak, using periodic boundary conditions.

    Returns
    -------
    r : ndarray
        Distance from the peak, starting at r=0.
    profile : ndarray
        Field values along the positive y-axis.
    """
    N = field.shape[0]
    dx = L / N

    ix, iy, iz = (int(i) for i in center)

    if r_max > L / 2:
        raise ValueError("r_max must not exceed L/2.")

    n_steps = int(np.floor(r_max / dx))
    offsets = np.arange(n_steps + 1)

    y_indices = (iy + offsets) % N

    r = offsets * dx / rstar
    profile = field[ix, y_indices, iz]
    return r, profile

def y_neg_axis_profile(field, L, center, r_max, rstar, physical=True):
    """
    Extract the field profile along the negative y-axis
    from a peak, using periodic boundary conditions.

    Returns
    -------
    r : ndarray
        Distance from the peak, starting at r=0.
    profile : ndarray
        Field values along the negative y-axis.
    """
    N = field.shape[0]
    dx = L / N

    ix, iy, iz = (int(i) for i in center)

    if r_max > L / 2:
        raise ValueError("r_max must not exceed L/2.")

    n_steps = int(np.floor(r_max / dx))
    offsets = np.arange(n_steps + 1)

    y_indices = (iy - offsets) % N

    r = offsets * dx / rstar
    profile = field[ix, y_indices, iz]
    return r, profile

def z_axis_profile(field, L, center, r_max, rstar, physical=True):
    """
    Extract the field profile along the positive z-axis
    from a peak, using periodic boundary conditions.

    Returns
    -------
    r : ndarray
        Distance from the peak, starting at r=0.
    profile : ndarray
        Field values along the positive z-axis.
    """
    N = field.shape[0]
    dx = L / N

    ix, iy, iz = (int(i) for i in center)

    if r_max > L / 2:
        raise ValueError("r_max must not exceed L/2.")

    n_steps = int(np.floor(r_max / dx))
    offsets = np.arange(n_steps + 1)

    z_indices = (iz + offsets) % N

    r = offsets * dx / rstar
    profile = field[ix, iy, z_indices]
    return r, profile

def z_neg_axis_profile(field, L, center, r_max, rstar, physical=True):
    """
    Extract the field profile along the negative z-axis
    from a peak, using periodic boundary conditions.

    Returns
    -------
    r : ndarray
        Distance from the peak, starting at r=0.
    profile : ndarray
        Field values along the negative z-axis.
    """
    N = field.shape[0]
    dx = L / N

    ix, iy, iz = (int(i) for i in center)

    if r_max > L / 2:
        raise ValueError("r_max must not exceed L/2.")

    n_steps = int(np.floor(r_max / dx))
    offsets = np.arange(n_steps + 1)

    z_indices = (iz - offsets) % N

    r = offsets * dx / rstar
    profile = field[ix, iy, z_indices]
    return r, profile

def main():
    if N < 2 or L <= 0 or N_REALIZATIONS < 1:
        raise ValueError("Invalid grid, box, realization count or bin count")
    if NORMALIZATION not in ("sample", "theory"):
        raise ValueError("NORMALIZATION must be 'sample' or 'theory'")

    sigma0, sigma1, sigma2, gamma, rstar = spectral_parameters()
    print(f"sigma_0={sigma0:.8g}, sigma_1={sigma1:.8g}, sigma_2={sigma2:.8g}")
    print(f"gamma={gamma:.8g}, R_star={rstar:.8g}")
    print(f"beta={BETA_NG:.8g}")

    text_file = open(filename + "_info.txt", "w")

    text_file.write(f"N={N}, L={L:.8g}, N_REALIZATIONS={N_REALIZATIONS}\n")
    text_file.write(f"sigma_0={sigma0:.8g}, sigma_1={sigma1:.8g}, sigma_2={sigma2:.8g}\n")
    text_file.write(f"gamma={gamma:.8g}, R_star={rstar:.8g}\n")
    text_file.write(f"beta={BETA_NG:.8g}\n")

    text_file.close()


    pk = fft_spectrum(N, L)
    sqrt_pk = np.sqrt(pk)
    rng = np.random.default_rng(SEED)
    total = 0
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.writer(handle)
        for realization in range(N_REALIZATIONS):
            field = generate_field(rng, sqrt_pk, N, L)
            rms = float(np.std(field)) if NORMALIZATION == "sample" else sigma0
            if rms <= 0 or not np.isfinite(rms):
                raise ValueError("Field RMS is zero or invalid; check spectrum/grid")
            normalized = field/rms
            peaks = find_peaks(normalized, PEAK_THRESHOLD, MIN_SEPARATION)
            write_common = True
            for index in peaks:
                rx, profile_xp = x_axis_profile(normalized, L, index, R_MAX_RSTAR * rstar, rstar)
                _, profile_xn = x_neg_axis_profile(normalized, L, index, R_MAX_RSTAR * rstar, rstar)
                _, profile_yp = y_axis_profile(normalized, L, index, R_MAX_RSTAR * rstar, rstar)
                _, profile_yn = y_neg_axis_profile(normalized, L, index, R_MAX_RSTAR * rstar, rstar)
                _, profile_zp = z_axis_profile(normalized, L, index, R_MAX_RSTAR * rstar, rstar)
                _, profile_zn = z_neg_axis_profile(normalized, L, index, R_MAX_RSTAR * rstar, rstar)
                if write_common:
                    writer.writerow(rx)
                writer.writerow(profile_xp)
                writer.writerow(profile_xn)
                writer.writerow(profile_yp)
                writer.writerow(profile_yn)
                writer.writerow(profile_zp)
                writer.writerow(profile_zn)
            total += len(peaks)
            print(f"Realization {realization+1}/{N_REALIZATIONS}: {len(peaks)} peaks")
    print(f"Saved {total} profiles to {OUTPUT}")


if __name__ == "__main__":
    main()
