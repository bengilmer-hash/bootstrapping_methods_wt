import numpy as np
from numpy.typing import ArrayLike
from scipy.fft import rfft, rfftfreq
import pycwt as wavelet
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import TypeAlias

MotherWavelet: TypeAlias = (
    wavelet.Morlet | wavelet.Paul | wavelet.DOG | wavelet.MexicanHat
)


def wavelet_power(signal: ArrayLike, dt: float, dj: float, s0: float, J: float, mother: MotherWavelet):
    '''
    Creates a wavelet power spectra utilizing the pycwt package

    Parameters:
    -----------
    signal : arraylike
        The data to be analyzed
    dt : float
        The spacing between measurements
    dj : float
        The spacing between wavelet scales in suboctaves per octave
    s0 : float
        The minimum scale value you are interested in, usually nyquist frequency
    J : float
        The maximum scale value you are interested in
    mother : MotherWavelet
        The wavelet cwt will use for the analysis, ex. wavelet.Morlet

    Returns:
    --------
    wave : ArrayLike
        Raw wavelet transform cooefficients
    scales : ArrayLike
        Scales used for the transform
    periods : ArrayLike
        Inverse of the fftfreqs for plotting
    coi : ArrayLike
        Cone of influence for plotting what of the power spectra is reliable
    fft : ArrayLike
        The fft of the signal
    power : ArrayLike
        The power at a given point calculated by squaring the modulus of the coefficient at a point and normalizing by the scale
    avg_power: ArrayLike
        Average power across a particular period
    '''
    wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(signal, dt, dj, s0, J, mother)
    periods = 1/freqs

    power = (np.abs(wave)**2) / scales[:,np.newaxis]
    avg_power = power.mean(axis=1)
    return wave, scales, periods, coi, fft, power, avg_power


def fft_scramble(signal):
    '''
    Creates a randomized version of the data by scrambling the phases of the fft
    '''

    fft = rfft(signal)
    magnitude = np.abs(fft)
    phases = np.angle(fft)
    N = len(signal)

    scrambled_phases = np.random.uniform(-np.pi, np.pi, len(phases))

    # Do not scramble the phase of the DC offset, otherwise it could become complex
    scrambled_phases[0] = 0

    scramble = magnitude * np.exp(1j * scrambled_phases)

    bootstrap = np.fft.irfft(scramble).real

    return bootstrap



def scramble_bootstrap(signal : ArrayLike, dt : float, dj : float, s0 : float, J : float, mother : MotherWavelet, num_bootstraps : int = 1000, alpha : float = .05, plot_last: bool = False):
    '''
    Calculates significance levels using a null hypothesis based on phase of any component frequencies in a signal should be uniformly distributed if there is nothing causing oscillatory behavior at a given time

    Doesn't plot unless plot_last == True and fig.show() is called on the fig object to prevent pre-plotting

    Parameters:
    -----------
    signal : arraylike
        The data to be analyzed
    dt : float
        The spacing between measurements
    dj : float
        The spacing between wavelet scales in suboctaves per octave
    s0 : float
        The minimum scale value you are interested in, usually nyquist frequency
    J : float
        The maximum scale value you are interested in
    mother : MotherWavelet
        The wavelet cwt will use for the analysis, ex. wavelet.Morlet
    num_bootstraps : int
        The number of bootstraps to be used, more bootstraps usually corresponds higher significance
    alpha : float
        Desired p-value or 1 - significance threshold
    plot_last : bool
        Whether or not the last scrambled signal should be plotted

    Returns:
    --------
    wave : ArrayLike
        Raw wavelet transform cooefficients
    scales : ArrayLike
        Scales used for the transform
    periods : ArrayLike
        Inverse of the fftfreqs for plotting
    coi : ArrayLike
        Cone of influence for plotting what of the power spectra is reliable
    fft : ArrayLike
        The fft of the signal
    power : ArrayLike
        The power at a given point calculated by squaring the modulus of the coefficient at a point and normalizing by the scale
    avg_power: ArrayLike
        Average power across a particular period
    p_values : ArrayLike
        Maps a p-value to every pixel in the wavelet power spectra
    fig : Figure
        Figure containing the plot of the last scrambled signal, call fig.show() to display
    '''

    wave, scales, periods, coi, fft, power, avg_power = wavelet_power(signal, dt, dj, s0, J, mother)

    extremes_count = np.zeros((len(scales), len(signal)))

    avg_powers = np.zeros((len(scales), num_bootstraps))
    
    for b in range(num_bootstraps):
        bootstrap = fft_scramble(signal)
        wave_s, scales_s, periods_s, fft_s, coi_s, power_s, avg_power_s = wavelet_power(bootstrap, dt, dj, s0, J, mother)
        extremes_count[:,:] += (power_s >= power)
        avg_powers[:,b] = avg_power_s

    
    p_values = extremes_count / num_bootstraps

    percentile_target = (1.0 - alpha) * 100
    sig_line = np.percentile(avg_powers, percentile_target, axis=1)

    if plot_last == True:
        with plt.ioff():
            t = range(len(signal)) * dt

            fig, ax = plt.subplot()
            ax.plot(t, bootstrap, label='Scrambled')
            ax.plot(t, signal, label='Signal')
            ax.set_title('Signal vs Scramble')
            ax.legend()
            return power, periods, coi, avg_power, p_values, sig_line, fig
    else:
        return power, periods, coi, avg_power, p_values, sig_line


    def wps_plot(power: Arraylike, t: Arraylike, periods: Arraylike, coi: Arraylike, avg_power: Arraylike, p_values: Arraylike, sigline: Arraylike, cmap: str = 'turbo', title: str = 'Wavelet Power Spectrum'):
        '''
        Plots the output of a wavelet power spectra function

        Parameters:
        -----------
        wave : ArrayLike
            Raw wavelet transform cooefficients
        scales : ArrayLike
            Scales used for the transform
        periods : ArrayLike
            Inverse of the fftfreqs for plotting
        coi : ArrayLike
            Cone of influence for plotting what of the power spectra is reliable
        fft : ArrayLike
            The fft of the signal
        power : ArrayLike
            The power at a given point calculated by squaring the modulus of the coefficient at a point and normalizing by the scale
        avg_power: ArrayLike
            Average power across a particular period
        p_values : ArrayLike
            Maps a p-value to every pixel in the wavelet power spectra
        sigline : ArrayLike
            Significance threshold at every period for plotting average power vs significance threshold
        cmap : str
            Matplotlib color map that should be used for display
        title : str
            Figure title
        
        Returns:
        --------
        fig : Figure
            Wavelet power spectra plot, call fig.show() to display
        '''
        with plt.ioff():
            fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True, 
                            gridspec_kw={'width_ratios': [4, 1]})
            fig.suptitle(title, fontsize=16)

            t_grid, p_grid = np.meshgrid(t, periods)

            # Wavelet power spectrum
            im = axs[0].pcolormesh(t_grid, p_grid, power, cmap=cmap, shading='auto')
            axs[0].contour(t_grid, p_grid, p_values, levels=[.01], colors='black')
            axs[0].set_title('Full Power Spectrum')
            axs[0].set_xlabel('Time (s)')
            axs[0].set_ylabel('Period (s)')
            axs[0].set_ylim(2 * dt, coi.max())

            # Add a colorbar to show the power levels
            fig.colorbar(im, ax=axs[0], label='Power', location='left')


            axs[0].fill_between(t, coi, periods.max(), color="w", alpha=0.5, hatch="x", label="COI")
            axs[0].plot(t, coi, "w--", lw=0.8, alpha=0.7)

            axs[0].legend(loc='upper right')

            
            # Averaged time spectra
            axs[1].plot(avg_power, periods)
            axs[1].plot(sigline, periods, 'k--', label='99% Significance')
            axs[1].set_title('Averaged Spectrum')
            axs[1].set_xlabel('Average Power')

            # Set the x-axis to a logarithmic scale to better see the peaks
            axs[1].set_xscale('log')
            axs[1].set_xlim(1e-4,1e2)
            axs[1].legend(loc='upper right')


            axs[0].invert_yaxis()
            axs[0].set_yscale('log')


            plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout for suptitle

        return fig
            