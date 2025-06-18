from flask import Flask, jsonify, request
from flask_cors import CORS
import subprocess
import sys

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow CORS from all origins

# Dictionary to store running processes
processes = {}

@app.route('/run/<script_name>', methods=['GET'])
def run_script(script_name):
    if script_name in processes:
        return jsonify({"message": f"{script_name} is already running"}), 400  # Prevent duplicate runs

    try:
        python_executable = sys.executable  # Gets the correct Python path
        process = subprocess.Popen([python_executable, f"{script_name}.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        processes[script_name] = process
        
        # Capture errors if the script fails
        output, error = process.communicate(timeout=10)  # Increased timeout value
        if error:
            print(f"❌ Error starting {script_name}: {error}")  # ✅ Debugging in terminal
            return jsonify({"error": error}), 500

        print(f"✅ {script_name} started successfully")
        return jsonify({"message": f"{script_name} started successfully", "output": output})
    
    except subprocess.TimeoutExpired:
        return jsonify({"message": f"{script_name} started successfully (running in background)"}), 200
    except Exception as e:
        print(f"❌ Unexpected error: {e}")  # ✅ Debugging in terminal
        return jsonify({"error": str(e)}), 500


@app.route('/stop/<script_name>', methods=['GET'])
def stop_script(script_name):
    if script_name not in processes:
        return jsonify({"message": f"{script_name} is not running"}), 400

    try:
        processes[script_name].terminate()
        del processes[script_name]
        return jsonify({"message": f"{script_name} stopped successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5500, debug=True)  # Changed port to 8080
