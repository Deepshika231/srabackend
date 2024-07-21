import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy import signal
from io import BytesIO
import base64

def load_data(file):
    data = pd.read_csv(file, header=None)  # Assuming no header
    profile = data.iloc[:, 0].values  # Assuming profile data is in the first column
    return profile

def calculate_parameters(profile):
    Ra = np.mean(np.abs(profile))
    Rq = np.sqrt(np.mean(profile**2))
    Sm = len(profile) / np.argmax(np.correlate(profile, profile, mode='full'))
    Rv = np.min(profile)
    Rp = np.max(profile)
    Rt = Rp - Rv

    # RMS Slope (assuming equally spaced points)
    dx = 1  # Assuming unit spacing
    rms_slope = np.sqrt(np.mean(np.gradient(profile, dx)**2))

    return Ra, Rq, Sm, rms_slope, Rv, Rp, Rt

def plot_profile(profile):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Plot P - waviness (filter profile)
    b, a = signal.butter(2, 0.1)
    waviness = signal.filtfilt(b, a, profile)
    axs[0, 0].plot(profile, label='Original Profile')
    axs[0, 0].plot(waviness, label='Waviness Profile')
    axs[0, 0].set_title('P - Waviness (Filter Profile)')
    axs[0, 0].legend()

    # Plot cumulative profile
    cumulative = np.cumsum(profile)
    axs[0, 1].plot(cumulative)
    axs[0, 1].set_title('Cumulative Profile')

    # Plot W - auto-correlation function
    auto_corr = np.correlate(profile, profile, mode='full') / len(profile)
    axs[1, 0].plot(auto_corr[auto_corr.size // 2:])
    axs[1, 0].set_title('W - Auto-correlation Function')

    # Plot R - power spectral density
    f, Pxx_den = welch(profile, nperseg=1024)
    axs[1, 1].semilogy(f, Pxx_den)
    axs[1, 1].set_title('R - Power Spectral Density')
    axs[1, 1].set_xlabel('Frequency [Hz]')
    axs[1, 1].set_ylabel('PSD [V**2/Hz]')

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return plot_data

def process_file(file):
    profile = load_data(file)
    Ra, Rq, Sm, rms_slope, Rv, Rp, Rt = calculate_parameters(profile)
    results = {
        'Ra': Ra,
        'Rq': Rq,
        'Sm': Sm,
        'rms_slope': rms_slope,
        'Rv': Rv,
        'Rp': Rp,
        'Rt': Rt
    }
    plot_data = plot_profile(profile)
    return results, plot_data
