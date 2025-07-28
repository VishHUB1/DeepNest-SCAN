import cv2
import numpy as np
import os
import time
import threading
import socket
import webbrowser
import requests
from io import BytesIO
from PIL import Image
from flask import Flask, Response, render_template_string, request, jsonify
from datetime import datetime

# Create Flask app
app = Flask(__name__)

# Global variables
esp32_url = "http://192.168.0.104/"  # Default ESP32-CAM URL
capture_interval = 5  # seconds
is_capturing = False
capture_thread = None
current_frame = None
current_capture = None
image_count = 0
save_directory = "C:/Users/visha/Desktop/Temp/drones_project/data collection/full data"

# Create save directory if it doesn't exist
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

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

# HTML template as a string for render_template_string
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ESP32-CAM Data Collection</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
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
            border: 1px solid #ccc;
            padding: 10px;
            border-radius: 5px;
        }
        h2 {
            margin-top: 0;
        }
        .video-feed {
            width: 100%;
            height: auto;
        }
        .controls {
            margin: 15px 0;
        }
        button {
            padding: 10px 15px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            margin-right: 10px;
        }
        button:hover {
            background-color: #0056b3;
        }
        input {
            padding: 8px;
            width: 250px;
            margin-right: 10px;
        }
        #status {
            font-style: italic;
            margin-top: 10px;
        }
        #imageCount {
            font-weight: bold;
            color: #007bff;
        }
        .error {
            color: red;
            font-weight: bold;
        }
        .success {
            color: green;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>ESP32-CAM Data Collection</h1>
        
        <div class="controls">
            <input type="text" id="ipAddressInput" placeholder="ESP32-CAM IP Address" value="{{ esp32_url }}">
            <button onclick="startCapture()">Start Stream</button>
            <button onclick="stopCapture()">Stop Capture</button>
            <button onclick="testConnection()">Test Connection</button>
        </div>
        
        <div id="status">Status: Ready to start capture</div>
        <div>Images captured: <span id="imageCount">0</span></div>
        <div>Save directory: <span id="saveDir">{{ save_directory }}</span></div>
        
        <div class="video-container">
            <div class="video-box">
                <h2>Live Stream</h2>
                <img src="/video_feed" class="video-feed" id="liveStream">
            </div>
            <div class="video-box">
                <h2>Captured Frame</h2>
                <img src="/captured_feed" class="video-feed">
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
        
        function updateImageCount(count) {
            document.getElementById('imageCount').textContent = count;
        }
        
        function startCapture() {
            const ipAddress = document.getElementById('ipAddressInput').value;
            updateStatus("Starting capture...");
            
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
            })
            .catch(error => {
                updateStatus("Error starting capture: " + error, true);
            });
        }
        
        function stopCapture() {
            fetch('/stop_capture')
            .then(response => response.json())
            .then(data => {
                updateStatus(data.status);
            })
            .catch(error => {
                updateStatus("Error stopping capture: " + error, true);
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
        
        // Update image count every second
        setInterval(() => {
            fetch('/get_image_count')
            .then(response => response.json())
            .then(data => {
                updateImageCount(data.count);
            });
        }, 1000);
    </script>
</body>
</html>
"""

# Function to capture frames from ESP32-CAM
def capture_frames():
    global current_frame, current_capture, image_count, is_capturing
    
    print(f"Starting capture from: {esp32_url}")
    
    # Different methods to try for stream capture
    methods = [
        # Method 1: Direct OpenCV VideoCapture
        lambda: direct_capture(),
        # Method 2: Using requests to get MJPEG stream
        lambda: mjpeg_capture(),
        # Method 3: Get individual JPEGs from ESP32
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
    global current_frame, current_capture, image_count, is_capturing
    
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
        
        # Check if it's time to capture a new frame
        current_time = time.time()
        if current_time - last_capture_time >= capture_interval:
            # Save the captured frame
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{save_directory}/capture_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            
            # Update the current capture
            current_capture = frame.copy()
            image_count += 1
            
            print(f"Captured frame saved as {filename}")
            last_capture_time = current_time
    
    cap.release()
    print("Direct capture stopped")

def mjpeg_capture():
    global current_frame, current_capture, image_count, is_capturing
    
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
                    
                    # Check if it's time to capture a new frame
                    current_time = time.time()
                    if current_time - last_capture_time >= capture_interval:
                        # Save the captured frame
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"{save_directory}/capture_{timestamp}.jpg"
                        cv2.imwrite(filename, frame)
                        
                        # Update the current capture
                        current_capture = frame.copy()
                        image_count += 1
                        
                        print(f"Captured frame saved as {filename}")
                        last_capture_time = current_time
    
    print("MJPEG capture stopped")

def jpeg_capture():
    global current_frame, current_capture, image_count, is_capturing
    
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
                    
                    # Check if it's time to capture a new frame
                    current_time = time.time()
                    if current_time - last_capture_time >= capture_interval:
                        # Save the captured frame
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"{save_directory}/capture_{timestamp}.jpg"
                        cv2.imwrite(filename, frame)
                        
                        # Update the current capture
                        current_capture = frame.copy()
                        image_count += 1
                        
                        print(f"Captured frame saved as {filename}")
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
    return render_template_string(html_template, esp32_url=esp32_url, save_directory=save_directory)

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

@app.route('/captured_feed')
def captured_feed():
    def generate():
        global current_capture
        while True:
            if current_capture is not None:
                ret, jpeg = cv2.imencode('.jpg', current_capture)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
            else:
                # Return a blank frame if no current capture
                blank = np.ones((480, 640, 3), np.uint8) * 200
                msg = "Waiting for first capture..."
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
        return jsonify({"status": "Capture already running", "error": False})
    
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
    
    return jsonify({"status": f"Capture started from {esp32_url}", "error": False})

@app.route('/stop_capture')
def stop_capture():
    global is_capturing
    
    if not is_capturing:
        return jsonify({"status": "No capture running", "error": False})
    
    is_capturing = False
    # Wait for the thread to finish
    if capture_thread and capture_thread.is_alive():
        capture_thread.join(timeout=1.0)
    
    return jsonify({"status": "Capture stopped", "error": False})

@app.route('/get_image_count')
def get_image_count():
    global image_count
    return jsonify({"count": image_count})

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
    
    print(f"\nESP32-CAM Data Collection App")
    print(f"-----------------------------")
    print(f"Server running at: http://{ip}:{port}")
    print(f"ESP32-CAM URL: {esp32_url}")
    print(f"Images will be saved to: {os.path.abspath(save_directory)}")
    print(f"Capture interval: {capture_interval} seconds")
    
    # Open web browser automatically
    webbrowser.open(f"http://{ip}:{port}")
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=port, debug=True)