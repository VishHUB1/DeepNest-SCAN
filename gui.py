import cv2
import numpy as np
import os
import time
import threading
import socket
import webbrowser
import requests
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
from flask import Flask, Response, render_template_string, request, jsonify

# Create Flask app
app = Flask(__name__)

# Global variables
esp32_url = "http://192.168.0.104/"  # Default ESP32-CAM URL
capture_interval = 2  # seconds
is_capturing = False
capture_thread = None
current_frame = None
current_processed_frame = None
image_count = 0
save_directory = "C:/Users/visha/Desktop/Temp/drones_project/realtimedata"
model = None
reconstruction_loss = 0.0
is_cracked = False

# Create save directory if it doesn't exist
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Function to load the pre-trained Keras model
def load_model():
    global model
    try:
        # Load your pre-trained Keras model
        print("Loading Keras model...")
        model = keras.models.load_model('best_model.keras')
        print("Model loaded successfully")
        
        # Check if model loaded correctly by running inference on a test image
        test_img = np.ones((224, 224, 3), dtype=np.float32)
        test_img = np.expand_dims(test_img, axis=0)
        _ = model.predict(test_img)
        print("Model verified with test prediction")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

# Function to preprocess image for the model
def preprocess_image(image):
    # Resize to the input size expected by your model
    # Adjust these parameters based on your model's requirements
    img_size = (224, 224)
    img = cv2.resize(image, img_size)
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to detect cracks in the image
def detect_crack(image):
    global reconstruction_loss, is_cracked
    if model is None:
        return image, 0.0, False
    
    try:
        # Preprocess the image
        processed_img = preprocess_image(image)
        
        # Get prediction from model
        # Assuming it's an autoencoder that returns reconstructed image and provides reconstruction loss
        reconstructed = model.predict(processed_img, verbose=0)
        
        # Calculate reconstruction loss (MSE)
        loss = np.mean(np.square(processed_img - reconstructed))
        reconstruction_loss = loss
        
        # Determine if cracked based on threshold
        # Adjust this threshold based on your model's performance
        threshold = 0.1
        is_cracked = loss > threshold
        
        # Create a copy of the image for visualization
        result_img = image.copy()
        
        # Add text showing results
        status = "CRACKED" if is_cracked else "NOT CRACKED"
        color = (0, 0, 255) if is_cracked else (0, 255, 0)
        cv2.putText(result_img, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2)
        cv2.putText(result_img, f"Loss: {loss:.6f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw border around the image based on result
        border_size = 5
        border_color = color
        h, w = result_img.shape[:2]
        result_img = cv2.copyMakeBorder(result_img, border_size, border_size, 
                                        border_size, border_size, 
                                        cv2.BORDER_CONSTANT, value=border_color)
        
        return result_img, loss, is_cracked
    
    except Exception as e:
        print(f"Error in crack detection: {e}")
        return image, 0.0, False

# Function to get local IP address
def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

# HTML template for the web interface
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ESP32-CAM Crack Detection</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .video-container {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }
        .video-box {
            flex: 1;
            min-width: 320px;
            background-color: white;
            border: 1px solid #ccc;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
        }
        h2 {
            margin-top: 0;
            color: #444;
            border-bottom: 1px solid #eee;
            padding-bottom: 8px;
        }
        .video-feed {
            width: 100%;
            height: auto;
            border-radius: 4px;
        }
        .controls {
            margin: 15px 0;
            background-color: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        button {
            padding: 10px 15px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            margin-right: 10px;
            transition: background-color 0.3s;
        }
        button:hover {
            background-color: #0056b3;
        }
        input {
            padding: 10px;
            width: 250px;
            margin-right: 10px;
            border: 1px solid #ccc;
            border-radius: 5px;
        }
        #status {
            font-style: italic;
            margin-top: 10px;
        }
        #crackStatus {
            font-weight: bold;
            font-size: 1.2em;
            margin-top: 10px;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            margin-bottom: 10px;
        }
        .cracked {
            background-color: #ffebee;
            color: #d32f2f;
        }
        .not-cracked {
            background-color: #e8f5e9;
            color: #388e3c;
        }
        .loss-value {
            font-size: 0.9em;
            color: #666;
            text-align: center;
            margin-bottom: 15px;
        }
        .info-panel {
            background-color: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .loading {
            text-align: center;
            padding: 20px;
            font-style: italic;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>ESP32-CAM Crack Detection</h1>
        
        <div class="controls">
            <input type="text" id="ipAddressInput" placeholder="ESP32-CAM IP Address" value="{{ esp32_url }}">
            <button onclick="startCapture()">Start Stream</button>
            <button onclick="stopCapture()">Stop Stream</button>
            <button onclick="testConnection()">Test Connection</button>
        </div>
        
        <div class="info-panel">
            <div id="status">Status: Ready to start capture</div>
            <div id="crackStatus" class="not-cracked">Result: Waiting for analysis</div>
            <div class="loss-value">Reconstruction Loss: <span id="lossValue">--</span></div>
        </div>
        
        <div class="video-container">
            <div class="video-box">
                <h2>Live Stream</h2>
                <img src="/video_feed" class="video-feed" id="liveStream">
            </div>
            <div class="video-box">
                <h2>Crack Analysis (Every {{ capture_interval }} sec)</h2>
                <img src="/processed_feed" class="video-feed" id="processedStream">
                <div class="loading" id="loadingIndicator">Waiting for first analysis...</div>
            </div>
        </div>
    </div>

    <script>
        function updateStatus(status, isError = false) {
            const statusEl = document.getElementById('status');
            statusEl.textContent = "Status: " + status;
            
            if (isError) {
                statusEl.className = "error";
            } else {
                statusEl.className = "";
            }
        }
        
        function updateCrackStatus(isCracked, loss) {
            const crackStatusEl = document.getElementById('crackStatus');
            const lossValueEl = document.getElementById('lossValue');
            const loadingIndicatorEl = document.getElementById('loadingIndicator');
            
            if (loss > 0) {
                loadingIndicatorEl.style.display = 'none';
            }
            
            if (isCracked) {
                crackStatusEl.textContent = "Result: CRACKED";
                crackStatusEl.className = "cracked";
            } else {
                crackStatusEl.textContent = "Result: NOT CRACKED";
                crackStatusEl.className = "not-cracked";
            }
            
            lossValueEl.textContent = loss.toFixed(6);
        }
        
        function startCapture() {
            const ipAddress = document.getElementById('ipAddressInput').value;
            updateStatus("Starting stream...");
            
            fetch('/start_capture', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    ip_address: ipAddress
                })
            })
            .then(response => response.json())
            .then(data => {
                updateStatus(data.status, data.error);
                
                // Refresh the stream src to ensure it's using the new URL
                document.getElementById('liveStream').src = "/video_feed?" + new Date().getTime();
                document.getElementById('processedStream').src = "/processed_feed?" + new Date().getTime();
            })
            .catch(error => {
                updateStatus("Error starting stream: " + error, true);
            });
        }
        
        function stopCapture() {
            fetch('/stop_capture')
            .then(response => response.json())
            .then(data => {
                updateStatus(data.status);
            })
            .catch(error => {
                updateStatus("Error stopping stream: " + error, true);
            });
        }
        
        function testConnection() {
            const ipAddress = document.getElementById('ipAddressInput').value;
            updateStatus("Testing connection...");
            
            fetch('/test_connection', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    ip_address: ipAddress
                })
            })
            .then(response => response.json())
            .then(data => {
                updateStatus(data.status, !data.success);
            })
            .catch(error => {
                updateStatus("Connection test failed: " + error, true);
            });
        }
        
        // Update crack status every second
        setInterval(() => {
            fetch('/get_crack_status')
            .then(response => response.json())
            .then(data => {
                updateCrackStatus(data.is_cracked, data.loss);
            });
        }, 1000);
    </script>
</body>
</html>
"""

# Function to capture frames from ESP32-CAM
def capture_frames():
    global current_frame, current_processed_frame, image_count, is_capturing
    
    print(f"Starting capture from: {esp32_url}")
    
    # Different methods to try for stream capture
    methods = [
        lambda: direct_capture(),
        lambda: mjpeg_capture(),
        lambda: jpeg_capture()
    ]
    
    # Try all methods until one works
    for method in methods:
        try:
            method()
            return
        except Exception as e:
            print(f"Method failed with error: {e}")
    
    print("All capture methods failed")
    is_capturing = False

def direct_capture():
    global current_frame, current_processed_frame, image_count, is_capturing
    
    # Try standard OpenCV capture
    stream_url = esp32_url
    if not stream_url.endswith('/'):
        stream_url += '/'
    
    cap = cv2.VideoCapture(stream_url)
    
    if not cap.isOpened():
        print(f"Error: Could not open video stream from {stream_url}")
        raise Exception("Failed to open video stream")
    
    last_capture_time = time.time()
    
    while is_capturing:
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to get frame, retrying...")
            time.sleep(0.5)
            continue
            
        current_frame = frame
        
        # Check if it's time to capture a new frame for processing
        current_time = time.time()
        if current_time - last_capture_time >= capture_interval:
            # Save the captured frame
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{save_directory}/capture_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            image_count += 1
            
            # Process the frame for crack detection
            processed_frame, _, _ = detect_crack(frame.copy())
            current_processed_frame = processed_frame
            
            print(f"Processed frame {image_count} - Saved as {filename}")
            last_capture_time = current_time
    
    cap.release()
    print("Direct capture stopped")

def mjpeg_capture():
    global current_frame, current_processed_frame, image_count, is_capturing
    
    # Try MJPEG stream capture
    stream_url = esp32_url
    if not stream_url.endswith('/'):
        stream_url += 'mjpeg/1'  # Common MJPEG endpoint for ESP32-CAM
    elif not 'mjpeg' in stream_url:
        stream_url += 'mjpeg/1'
    
    print(f"Trying MJPEG stream at: {stream_url}")
    
    last_capture_time = time.time()
    
    # Open stream with requests
    r = requests.get(stream_url, stream=True, timeout=5)
    if r.status_code != 200:
        raise Exception(f"Failed to connect to MJPEG stream, status code: {r.status_code}")
    
    bytes_buffer = bytes()
    while is_capturing:
        # Read from the stream
        for chunk in r.iter_content(chunk_size=1024):
            if not is_capturing:
                break
                
            if not chunk:
                continue
                
            bytes_buffer += chunk
            
            # Check for JPEG markers
            a = bytes_buffer.find(b'\xff\xd8')  # JPEG start
            b = bytes_buffer.find(b'\xff\xd9')  # JPEG end
            
            if a != -1 and b != -1:
                jpg = bytes_buffer[a:b+2]
                bytes_buffer = bytes_buffer[b+2:]
                
                # Decode the image
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                
                if frame is not None:
                    current_frame = frame
                    
                    # Check if it's time to capture a new frame for processing
                    current_time = time.time()
                    if current_time - last_capture_time >= capture_interval:
                        # Save the captured frame
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"{save_directory}/capture_{timestamp}.jpg"
                        cv2.imwrite(filename, frame)
                        image_count += 1
                        
                        # Process the frame for crack detection
                        processed_frame, _, _ = detect_crack(frame.copy())
                        current_processed_frame = processed_frame
                        
                        print(f"Processed frame {image_count} - Saved as {filename}")
                        last_capture_time = current_time
    
    print("MJPEG capture stopped")

def jpeg_capture():
    global current_frame, current_processed_frame, image_count, is_capturing
    
    # Try getting static JPEG images
    capture_url = esp32_url
    if not capture_url.endswith('/'):
        capture_url += '/'
    
    # Common endpoints for ESP32-CAM
    endpoints = ['capture', 'jpg', 'snapshot.jpg', 'photo.jpg', '']
    
    # Find working endpoint
    working_url = None
    for endpoint in endpoints:
        test_url = capture_url + endpoint
        try:
            response = requests.get(test_url, timeout=2)
            if response.status_code == 200 and response.content.startswith(b'\xff\xd8'):
                working_url = test_url
                print(f"Found working endpoint: {working_url}")
                break
        except:
            continue
    
    if not working_url:
        raise Exception("Could not find working JPEG endpoint")
    
    last_capture_time = time.time()
    
    while is_capturing:
        try:
            # Get a new image
            response = requests.get(working_url, timeout=2)
            
            if response.status_code == 200 and response.content.startswith(b'\xff\xd8'):
                # Decode the image
                frame = cv2.imdecode(np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_COLOR)
                
                if frame is not None:
                    current_frame = frame
                    
                    # Check if it's time to capture a new frame for processing
                    current_time = time.time()
                    if current_time - last_capture_time >= capture_interval:
                        # Save the captured frame
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"{save_directory}/capture_{timestamp}.jpg"
                        cv2.imwrite(filename, frame)
                        image_count += 1
                        
                        # Process the frame for crack detection
                        processed_frame, _, _ = detect_crack(frame.copy())
                        current_processed_frame = processed_frame
                        
                        print(f"Processed frame {image_count} - Saved as {filename}")
                        last_capture_time = current_time
            
            # Wait a bit before getting the next frame
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error getting JPEG: {e}")
            time.sleep(1)
    
    print("JPEG capture stopped")

# Routes for the web interface
@app.route('/')
def index():
    return render_template_string(html_template, esp32_url=esp32_url, capture_interval=capture_interval)

@app.route('/video_feed')
def video_feed():
    def generate():
        global current_frame
        while True:
            if current_frame is not None:
                ret, jpeg = cv2.imencode('.jpg', current_frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
            else:
                # Return a blank frame if no current frame
                blank = np.ones((480, 640, 3), np.uint8) * 200
                msg = "Waiting for stream..."
                cv2.putText(blank, msg, (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                ret, jpeg = cv2.imencode('.jpg', blank)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
            time.sleep(0.1)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/processed_feed')
def processed_feed():
    def generate():
        global current_processed_frame
        while True:
            if current_processed_frame is not None:
                ret, jpeg = cv2.imencode('.jpg', current_processed_frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
            else:
                # Return a blank frame if no current processed frame
                blank = np.ones((480, 640, 3), np.uint8) * 200
                msg = "Waiting for first analysis..."
                cv2.putText(blank, msg, (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                ret, jpeg = cv2.imencode('.jpg', blank)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
            time.sleep(0.1)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_capture', methods=['POST'])
def start_capture():
    global is_capturing, capture_thread, esp32_url
    
    if is_capturing:
        return jsonify({"status": "Stream already running", "error": False})
    
    data = request.get_json()
    if data and 'ip_address' in data:
        esp32_url = data['ip_address']
        # Make sure URL starts with http:// if not provided
        if not esp32_url.startswith('http://') and not esp32_url.startswith('https://'):
            esp32_url = 'http://' + esp32_url
    
    is_capturing = True
    capture_thread = threading.Thread(target=capture_frames)
    capture_thread.daemon = True
    capture_thread.start()
    
    return jsonify({"status": f"Stream started from {esp32_url}", "error": False})

@app.route('/stop_capture')
def stop_capture():
    global is_capturing
    
    if not is_capturing:
        return jsonify({"status": "No stream running", "error": False})
    
    is_capturing = False
    # Wait for the thread to finish
    if capture_thread and capture_thread.is_alive():
        capture_thread.join(timeout=1.0)
    
    return jsonify({"status": "Stream stopped", "error": False})

@app.route('/get_crack_status')
def get_crack_status():
    global reconstruction_loss, is_cracked
    return jsonify({"loss": float(reconstruction_loss), "is_cracked": bool(is_cracked)})

@app.route('/test_connection', methods=['POST'])
def test_connection():
    data = request.get_json()
    if not data or 'ip_address' not in data:
        return jsonify({"status": "No IP address provided", "success": False})
    
    test_url = data['ip_address']
    if not test_url.startswith('http://') and not test_url.startswith('https://'):
        test_url = 'http://' + test_url
    
    try:
        # Try to connect to the ESP32-CAM
        response = requests.get(test_url, timeout=3)
        if response.status_code == 200:
            return jsonify({"status": f"Successfully connected to {test_url}", "success": True})
        else:
            return jsonify({"status": f"Connection failed with status code: {response.status_code}", "success": False})
    except requests.exceptions.RequestException as e:
        return jsonify({"status": f"Connection error: {str(e)}", "success": False})

# Main function
if __name__ == "__main__":
    ip = get_local_ip()
    port = 5000
    
    print(f"\nESP32-CAM Crack Detection App")
    print(f"-----------------------------")
    print(f"Server running at: http://{ip}:{port}")
    print(f"ESP32-CAM URL: {esp32_url}")
    print(f"Images will be saved to: {os.path.abspath(save_directory)}")
    print(f"Capture interval: {capture_interval} seconds")
    
    # Load the Keras model
    load_model()
    
    # Open web browser automatically
    webbrowser.open(f"http://{ip}:{port}")
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=port, debug=False)