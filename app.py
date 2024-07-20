import matplotlib
matplotlib.use('Agg')  # Use the non-interactive Agg backend

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy import signal
import base64
import matplotlib.pyplot as plt
from io import BytesIO

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/process', methods=['POST'])
def process_file():
    print(request.files)  # Log request files for debugging
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        try:
            # Read the file into a DataFrame
            data = pd.read_csv(file, header=None)
            profile = data.iloc[:, 0].values  # Assuming profile data is in the first column
            
            # Calculate parameters
            Ra = np.mean(np.abs(profile))
            Rq = np.sqrt(np.mean(profile**2))
            Sm = len(profile) / np.argmax(np.correlate(profile, profile, mode='full'))
            Rv = np.min(profile)
            Rp = np.max(profile)
            Rt = Rp - Rv
            dx = 1  # Assuming unit spacing
            rms_slope = np.sqrt(np.mean(np.gradient(profile, dx)**2))
            
            # Generate plot
            fig, axs = plt.subplots(2, 2, figsize=(10, 8))
            b, a = signal.butter(2, 0.1)
            waviness = signal.filtfilt(b, a, profile)
            axs[0, 0].plot(profile, label='Original Profile')
            axs[0, 0].plot(waviness, label='Waviness Profile')
            axs[0, 0].set_title('P - Waviness (Filter Profile)')
            axs[0, 0].legend()

            cumulative = np.cumsum(profile)
            axs[0, 1].plot(cumulative)
            axs[0, 1].set_title('Cumulative Profile')

            auto_corr = np.correlate(profile, profile, mode='full') / len(profile)
            axs[1, 0].plot(auto_corr[auto_corr.size // 2:])
            axs[1, 0].set_title('W - Auto-correlation Function')

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

            results = {
                'Ra': Ra,
                'Rq': Rq,
                'Sm': Sm,
                'rms_slope': rms_slope,
                'Rv': Rv,
                'Rp': Rp,
                'Rt': Rt,
                'plot': plot_data
            }

            return jsonify(results)

        except Exception as e:
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
