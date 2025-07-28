import cv2
import numpy as np
import os
import time
import threading
import socket
import webbrowser
import requests
from datetime import datetime
from flask import Flask, Response, render_template_string, request, jsonify
import tensorflow as tf
from collections import deque

# Create Flask app
app = Flask(__name__)

# Global variables
camera_url = "http://192.168.0.104/"  # Default IP camera URL
capture_interval = 2  # seconds (changed from 5 to 2 as requested)
is_streaming = False
stream_thread = None
current_frame = None
current_processed_frame = None
image_count = 0
save_directory = "C:/Users/visha/Desktop/Temp/drones_project/realtimedata"
model = None
recent_frames = deque(maxlen=5)  # Store the 5 most recent frames
reconstruction_losses = deque(maxlen=5)  # Store the 5 most recent reconstruction losses

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

# Function to load the Keras model
def load_model():
    global model
    try:
        # Load the model (replace with your actual model path)
        model = tf.keras.models.load_model("C:/Users/visha/Desktop/Temp/drones_project/results1/model/best_model.keras")
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

# Function to process frame with the model
def process_frame(frame):
    global model
    
    if model is None:
        # If model isn't loaded, create a placeholder result
        processed_frame = frame.copy()
        cv2.putText(processed_frame, "Model not loaded", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return processed_frame, "Unknown", 0.0
    
    try:
        # Preprocess the image for the model
        input_size = (224, 224)  # Adjust according to your model's input size
        img = cv2.resize(frame, input_size)
        img = img / 255.0  # Normalize
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        
        # Get predictions from model
        # Assuming it's an autoencoder model that returns the reconstructed image
        reconstructed = model.predict(img)
        
        # Calculate reconstruction loss (MSE)
        reconstruction_loss = np.mean(np.square(img - reconstructed))
        
        # Determine if cracked based on reconstruction loss threshold
        # You'll need to adjust this threshold based on your model
        threshold = 0.05  # Example threshold
        label = "Cracked" if reconstruction_loss > threshold else "Non-Cracked"
        
        # Create processed frame with results
        processed_frame = frame.copy()
        color = (0, 0, 255) if label == "Cracked" else (0, 255, 0)
        # Make text larger and more prominent
        cv2.putText(processed_frame, f"Status: {label}", (50, 50), 
           cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(processed_frame, f"Loss: {reconstruction_loss:.6f}", (50, 100), 
           cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        return processed_frame, label, reconstruction_loss
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        processed_frame = frame.copy()
        cv2.putText(processed_frame, "Processing Error", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return processed_frame, "Error", 0.0

# Function to capture frames from IP camera
def capture_frames():
    global current_frame, current_processed_frame, image_count, is_streaming, recent_frames, reconstruction_losses
    
    print(f"Starting stream from: {camera_url}")
    
    # Different methods to try for stream capture
    methods = [
        # Method 1: Direct OpenCV VideoCapture
        lambda: direct_capture(),
        # Method 2: Using requests to get MJPEG stream
        lambda: mjpeg_capture(),
        # Method 3: Get individual JPEGs
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
    is_streaming = False

def direct_capture():
    global current_frame, current_processed_frame, image_count, is_streaming, recent_frames, reconstruction_losses
    
    # Try standard OpenCV capture
    stream_url = camera_url
    if not stream_url.endswith('/'):
        stream_url += '/'
    
    cap = cv2.VideoCapture(stream_url)
    
    if not cap.isOpened():
        print(f"Error: Could not open video stream from {stream_url}")
        raise Exception("Failed to open video stream")
    
    last_capture_time = time.time()
    
    while is_streaming:
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to get frame, retrying...")
            time.sleep(0.5)
            continue
            
        current_frame = frame
        
        # Check if it's time to capture and process a new frame
        current_time = time.time()
        if current_time - last_capture_time >= capture_interval:
            # Process the frame with the model
            processed_frame, label, loss = process_frame(frame)
            current_processed_frame = processed_frame
            
            # Save the captured frame
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{save_directory}/capture_{timestamp}_{label}_{loss:.6f}.jpg"
            cv2.imwrite(filename, frame)
            
            # Add to recent frames and losses
            recent_frames.append((frame.copy(), timestamp, label))
            reconstruction_losses.append(loss)
            
            image_count += 1
            
            print(f"Captured frame {image_count}: {label}, Loss: {loss:.6f}, Saved as {filename}")
            last_capture_time = current_time
    
    cap.release()
    print("Direct capture stopped")

def mjpeg_capture():
    global current_frame, current_processed_frame, image_count, is_streaming, recent_frames, reconstruction_losses
    
    # Try MJPEG stream capture
    stream_url = camera_url
    if not stream_url.endswith('/'):
        stream_url += 'mjpeg/1'  # Common MJPEG endpoint
    elif not 'mjpeg' in stream_url:
        stream_url += 'mjpeg/1'
    
    print(f"Trying MJPEG stream at: {stream_url}")
    
    last_capture_time = time.time()
    
    # Open stream with requests
    r = requests.get(stream_url, stream=True, timeout=5)
    if r.status_code != 200:
        raise Exception(f"Failed to connect to MJPEG stream, status code: {r.status_code}")
    
    bytes_buffer = bytes()
    while is_streaming:
        # Read from the stream
        for chunk in r.iter_content(chunk_size=1024):
            if not is_streaming:
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
                    
                    # Check if it's time to capture and process a new frame
                    current_time = time.time()
                    if current_time - last_capture_time >= capture_interval:
                        # Process the frame with the model
                        processed_frame, label, loss = process_frame(frame)
                        current_processed_frame = processed_frame
                        
                        # Save the captured frame
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"{save_directory}/capture_{timestamp}_{label}_{loss:.6f}.jpg"
                        cv2.imwrite(filename, frame)
                        
                        # Add to recent frames and losses
                        recent_frames.append((frame.copy(), timestamp, label))
                        reconstruction_losses.append(loss)
                        
                        image_count += 1
                        
                        print(f"Captured frame {image_count}: {label}, Loss: {loss:.6f}, Saved as {filename}")
                        last_capture_time = current_time
    
    print("MJPEG capture stopped")

def jpeg_capture():
    global current_frame, current_processed_frame, image_count, is_streaming, recent_frames, reconstruction_losses
    
    # Try getting static JPEG images
    capture_url = camera_url
    if not capture_url.endswith('/'):
        capture_url += '/'
    
    # Common endpoints for IP cameras
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
    
    while is_streaming:
        try:
            # Get a new image
            response = requests.get(working_url, timeout=2)
            
            if response.status_code == 200 and response.content.startswith(b'\xff\xd8'):
                # Decode the image
                frame = cv2.imdecode(np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_COLOR)
                
                if frame is not None:
                    current_frame = frame
                    
                    # Check if it's time to capture and process a new frame
                    current_time = time.time()
                    if current_time - last_capture_time >= capture_interval:
                        # Process the frame with the model
                        processed_frame, label, loss = process_frame(frame)
                        current_processed_frame = processed_frame
                        
                        # Save the captured frame
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"{save_directory}/capture_{timestamp}_{label}_{loss:.6f}.jpg"
                        cv2.imwrite(filename, frame)
                        
                        # Add to recent frames and losses
                        recent_frames.append((frame.copy(), timestamp, label))
                        reconstruction_losses.append(loss)
                        
                        image_count += 1
                        
                        print(f"Captured frame {image_count}: {label}, Loss: {loss:.6f}, Saved as {filename}")
                        last_capture_time = current_time
            
            # Wait a bit before getting the next frame
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error getting JPEG: {e}")
            time.sleep(1)
    
    print("JPEG capture stopped")

# HTML template for the web interface
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Crack Detection System</title>
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
            margin-bottom: 20px;
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
        button.stop {
            background-color: #dc3545;
        }
        button.stop:hover {
            background-color: #c82333;
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
        .cracked {
            color: red;
            font-weight: bold;
        }
        .non-cracked {
            color: green;
            font-weight: bold;
        }
        .severity-high {
            background-color: #ffcccc;
        }
        .severity-medium {
            background-color: #ffffcc;
        }
        .severity-low {
            background-color: #ccffcc;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Crack Detection System</h1>
        
        <div class="controls">
            <input type="text" id="ipAddressInput" placeholder="IP Camera Address" value="{{ camera_url }}">
            <button onclick="startStream()">Start Stream</button>
            <button onclick="stopStream()" class="stop">Stop Stream</button>
            <button onclick="testConnection()">Test Connection</button>
        </div>
        
        <div id="status">Status: Ready to start streaming</div>
        <div>Frames captured: <span id="imageCount">0</span></div>
        <div>Save directory: <span id="saveDir">{{ save_directory }}</span></div>
        
        <div class="video-container">
            <div class="video-box">
                <h2>Live Stream</h2>
                <img src="/video_feed" class="video-feed" id="liveStream">
            </div>
            <div class="video-box">
                <h2>Processed Frame</h2>
                <img src="/processed_feed" class="video-feed">
            </div>
        </div>
        
        <div class="video-box">
            <h2>Recent Frames & Crack Severity Analysis</h2>
            <img src="/history_feed" class="video-feed">
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
        
        function startStream() {
            const ipAddress = document.getElementById('ipAddressInput').value;
            updateStatus("Starting stream...");
            
            fetch('/start_stream', {
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
                updateStatus("Error starting stream: " + error, true);
            });
        }
        
        function stopStream() {
            fetch('/stop_stream')
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

# Routes for the web interface
@app.route('/')
def index():
    return render_template_string(html_template, camera_url=camera_url, save_directory=save_directory)

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
                msg = "Waiting for first processed frame..."
                cv2.putText(blank, msg, (80, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                ret, jpeg = cv2.imencode('.jpg', blank)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
            time.sleep(0.1)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/history_feed')
def history_feed():
    def generate():
        global recent_frames, reconstruction_losses
        while True:
            if len(recent_frames) > 0:
                # Create a visual history display
                width = 640
                height = 480
                # Create a larger canvas for multiple frames
                canvas = np.ones((height, width, 3), dtype=np.uint8) * 200
                
                # Calculate how many frames we have
                num_frames = len(recent_frames)
                
                if num_frames > 0:
                    # Draw recent frames
                    frame_width = width // min(num_frames, 5)
                    frame_height = frame_width * 3 // 4
                    
                    for i, (frame, timestamp, label) in enumerate(recent_frames):
                        if i >= 5:  # Only show the 5 most recent
                            break
                        
                        # Resize frame to fit
                        resized_frame = cv2.resize(frame, (frame_width, frame_height))
                        
                        # Calculate position
                        x = i * frame_width
                        y = 30
                        
                        # Place frame on canvas
                        canvas[y:y+frame_height, x:x+frame_width] = resized_frame
                        
                        # Add label
                        label_color = (0, 0, 255) if label == "Cracked" else (0, 255, 0)
                        cv2.putText(canvas, label, (x+5, y+frame_height+20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1)
                        
                        # Add timestamp
                        short_timestamp = timestamp.split('_')[1]  # Just get the time part
                        cv2.putText(canvas, short_timestamp, (x+5, y+frame_height+40), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                    
                    # Add reconstruction loss and severity graph
                    if len(reconstruction_losses) > 0:
                        # Draw a severity graph
                        graph_y = frame_height + 70
                        graph_height = 100
                        
                        # Draw baseline
                        cv2.line(canvas, (0, graph_y + graph_height), (width, graph_y + graph_height), (0, 0, 0), 1)
                        
                        # Find max loss for scaling
                        max_loss = max(max(reconstruction_losses), 0.1)  # Ensure non-zero
                        
                        # Draw bars
                        bar_width = width // len(reconstruction_losses)
                        for i, loss in enumerate(reconstruction_losses):
                            # Calculate bar height
                            bar_height = int((loss / max_loss) * graph_height)
                            
                            # Define severity color
                            if loss > 0.1:  # High severity
                                color = (0, 0, 255)  # Red
                                severity = "High"
                            elif loss > 0.05:  # Medium severity
                                color = (0, 255, 255)  # Yellow
                                severity = "Medium"
                            else:  # Low severity
                                color = (0, 255, 0)  # Green
                                severity = "Low"
                                
                            # Draw bar
                            cv2.rectangle(canvas, 
                                         (i * bar_width, graph_y + graph_height - bar_height),
                                         ((i+1) * bar_width - 2, graph_y + graph_height),
                                         color, -1)
                            
                            # Add loss value
                            cv2.putText(canvas, f"{loss:.6f}", 
                                       (i * bar_width + 5, graph_y + graph_height - bar_height - 5),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                            
                            # Add severity label
                            cv2.putText(canvas, severity, 
                                       (i * bar_width + 5, graph_y + graph_height + 15),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                        
                        # Add title
                        cv2.putText(canvas, "Crack Severity Analysis (Reconstruction Loss)", 
                                   (20, graph_y - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                
                # Convert to JPEG
                ret, jpeg = cv2.imencode('.jpg', canvas)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
            else:
                # Return a blank canvas
                canvas = np.ones((480, 640, 3), dtype=np.uint8) * 200
                msg = "Waiting for captured frames..."
                cv2.putText(canvas, msg, (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                ret, jpeg = cv2.imencode('.jpg', canvas)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
            time.sleep(0.1)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_stream', methods=['POST'])
def start_stream():
    global is_streaming, stream_thread, camera_url
    
    if is_streaming:
        return jsonify({"status": "Stream already running", "error": False})
    
    data = request.get_json()
    if data and 'ip_address' in data:
        camera_url = data['ip_address']
        # Make sure URL starts with http:// if not provided
        if not camera_url.startswith('http://') and not camera_url.startswith('https://'):
            camera_url = 'http://' + camera_url
    
    # Load the model if not already loaded
    if model is None:
        load_model()
    
    is_streaming = True
    stream_thread = threading.Thread(target=capture_frames)
    stream_thread.daemon = True
    stream_thread.start()
    
    return jsonify({"status": f"Stream started from {camera_url}", "error": False})

@app.route('/stop_stream')
def stop_stream():
    global is_streaming
    
    if not is_streaming:
        return jsonify({"status": "No stream running", "error": False})
    
    is_streaming = False
    # Wait for the thread to finish
    if stream_thread and stream_thread.is_alive():
        stream_thread.join(timeout=1.0)
    
    return jsonify({"status": "Stream stopped", "error": False})

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
        # Try to connect to the IP camera
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
    
    print(f"\nCrack Detection System")
    print(f"---------------------")
    print(f"Server running at: http://{ip}:{port}")
    print(f"Camera URL: {camera_url}")
    print(f"Images will be saved to: {os.path.abspath(save_directory)}")
    print(f"Capture interval: {capture_interval} seconds")
    print(f"Loading model...")
    
    # Load the Keras model
    load_model()
    
    # Open web browser automatically
    webbrowser.open(f"http://{ip}:{port}")
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=port, debug=True)