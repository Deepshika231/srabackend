import matplotlib
matplotlib.use('Agg')  # Use the non-interactive Agg backend

from flask import Flask, request, jsonify
from flask_cors import CORS
from main1 import process_file  # Ensure this imports the correct functions

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/process', methods=['POST'])
def process_file_endpoint():
    print(request.files)  # Log request files for debugging
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        try:
            results, plot_data = process_file(file)

            return jsonify({
                'results': results,
                'plot': plot_data
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
