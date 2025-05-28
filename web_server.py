from flask import Flask, Response, send_from_directory, jsonify, render_template_string, url_for
import cv2
import os
from datetime import datetime

app = Flask(__name__)
video_path = 'output_video.avi'  # Or use 'My Video2.mp4' to show live detection feed

# Updated HTML template with modal, zoom, and violation type display
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>License Plate Violations</title>
  <style>
   :root {
  --bg: #f8fafc;
  --card: #ffffff;
  --primary: #1e293b;
  --secondary: #64748b;
  --highlight: #f1f5f9;
  --accent: #3b82f6;
  --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
  --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
  --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
  --radius-sm: 4px;
  --radius-md: 8px;
  --radius-lg: 12px;
}

* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

body {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  line-height: 1.5;
  margin: 0;
  padding: 0;
  background-color: var(--bg);
  color: var(--primary);
  max-width: 1800px;
  margin-inline: auto;
  padding: 2rem;
}

h1 {
  text-align: center;
  font-size: 2.5rem;
  font-weight: 700;
  margin-bottom: 3rem;
  color: var(--primary);
  letter-spacing: -0.025em;
}

.violations-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
  gap: 1.5rem;
}

.violation {
  background-color: var(--card);
  border-radius: var(--radius-lg);
  overflow: hidden;
  box-shadow: var(--shadow-md);
  display: flex;
  flex-direction: column;
  transition: all 0.2s ease;
  border: 1px solid rgba(0, 0, 0, 0.05);
}

.violation:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-lg);
}

.violation-main-image {
  width: 100%;
  height: 240px;
  object-fit: cover;
  background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
  border-bottom: 1px solid rgba(0, 0, 0, 0.05);
  cursor: pointer;
  transition: transform 0.2s ease;
}

.violation-main-image:hover {
  transform: scale(1.05);
}

.plate-images {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  padding: 0.75rem;
  background: var(--highlight);
  border-top: 1px solid rgba(0, 0, 0, 0.05);
  border-bottom: 1px solid rgba(0, 0, 0, 0.05);
}

.plate-image {
  height: 60px;
  border-radius: var(--radius-sm);
  border: 1px solid rgba(0, 0, 0, 0.1);
  background: #fff;
  transition: transform 0.2s ease;
}

.plate-image:hover {
  transform: scale(1.05);
  box-shadow: var(--shadow-sm);
}

.violation-info {
  padding: 1.25rem;
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.violation-info h3 {
  font-size: 1.125rem;
  font-weight: 600;
  margin: 0;
  color: var(--primary);
}

.violation-info p {
  font-size: 0.875rem;
  color: var(--secondary);
  margin: 0;
}

.violation-info .violation-type {
  font-weight: 700;
  color: var(--accent);
}

.plate-text {
  margin-top: 0.75rem;
  font-size: 0.95rem;
  font-weight: 600;
  padding: 0.75rem 1rem;
  border-radius: var(--radius-md);
  background: var(--highlight);
  color: var(--primary);
  font-family: 'Roboto Mono', monospace;
  border-left: 3px solid var(--accent);
}

.no-violations {
  grid-column: 1 / -1;
  text-align: center;
  color: #94a3b8;
  padding: 5rem 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}

.no-violations h2 {
  margin-bottom: 1rem;
  font-size: 1.75rem;
  font-weight: 600;
  color: var(--primary);
}

.no-violations p {
  max-width: 400px;
  margin: 0 auto;
  color: var(--secondary);
}

/* Modal styles */
.modal {
  display: none; /* Hidden by default */
  position: fixed;
  z-index: 1000;
  padding-top: 60px;
  left: 0;
  top: 0;
  width: 100%;
  height: 100%;
  overflow: auto;
  background-color: rgba(0,0,0,0.9);
}

.modal-content {
  margin: auto;
  display: block;
  max-width: 90%;
  max-height: 90%;
  transition: transform 0.25s ease;
  cursor: zoom-in;
}

.modal-content.zoomed {
  transform: scale(2); /* Zoom scale */
  cursor: zoom-out;
}

.close {
  position: absolute;
  top: 30px;
  right: 35px;
  color: #fff;
  font-size: 40px;
  font-weight: bold;
  cursor: pointer;
  user-select: none;
  z-index: 1100;
}

/* Responsive adjustments for modal image */
@media (max-width: 768px) {
  .modal-content {
    max-width: 100%;
    max-height: 80%;
  }
}

/* Scrollbar styling */
::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

::-webkit-scrollbar-track {
  background: var(--bg);
}

::-webkit-scrollbar-thumb {
  background: #cbd5e1;
  border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
  background: #94a3b8;
}

/* Responsive adjustments */
@media (max-width: 768px) {
  body {
    padding: 1rem;
  }
  
  h1 {
    font-size: 2rem;
    margin-bottom: 2rem;
  }
  
  .violations-grid {
    grid-template-columns: 1fr;
  }
  
  .violation-main-image {
    height: 200px;
  }
}
  </style>
</head>
<body>
  <h1>License Plate Violations</h1>
  
  {% if violations %}
  <div class="violations-grid">
    {% for violation in violations %}
    <div class="violation">
      <img src="{{ url_for('get_violation_image', filename=violation.filename) }}" 
           class="violation-main-image clickable-image"
           alt="Violation {{ loop.index }}"
           onclick="openModal(this.src)">
      {% if violation.plates %}
      <div class="plate-images">
        {% for plate in violation.plates %}
        <img src="{{ url_for('get_plate_image', filename=plate.filename) }}" 
             class="plate-image clickable-image"
             alt="Detected plate"
             onclick="openModal(this.src)">
        {% endfor %}
      </div>
      {% endif %}
      <div class="violation-info">
        <h3>Violation #{{ loop.index }}</h3>
        <p class="violation-type">Type: {{ violation.violation_type }}</p>
        <p>Detected: {{ violation.timestamp }}</p>
        <p>File: {{ violation.filename }}</p>
        {% if violation.plate_text %}
        <div class="plate-text">
          Detected Plate: {{ violation.plate_text }}
        </div>
        {% endif %}
      </div>
    </div>
    {% endfor %}
  </div>
  {% else %}
  <div class="no-violations">
    <h2>No violations detected yet</h2>
    <p>Check back later for new results.</p>
  </div>
  {% endif %}

  <!-- Modal for showing clicked image -->
  <div id="imageModal" class="modal">
    <span class="close" id="modalClose">&times;</span>
    <img class="modal-content" id="modalImg" alt="Zoomed Violation Image"/>
  </div>

  <script>
    const modal = document.getElementById('imageModal');
    const modalImg = document.getElementById('modalImg');
    const modalClose = document.getElementById('modalClose');

    function openModal(src) {
      modal.style.display = "block";
      modalImg.src = src;
      modalImg.classList.remove('zoomed');
    }

    modalClose.onclick = function() {
      modal.style.display = "none";
    }

    modalImg.onclick = function() {
      if (modalImg.classList.contains('zoomed')) {
        modalImg.classList.remove('zoomed');
      } else {
        modalImg.classList.add('zoomed');
      }
    }

    // Close modal if clicked outside image
    modal.onclick = function(event) {
      if (event.target == modal) {
        modal.style.display = "none";
      }
    }
  </script>
</body>
</html>
'''

# Endpoint to stream video
@app.route('/video')
def video_feed():
    def generate():
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        cap.release()
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Endpoint to fetch plate info
@app.route('/api/plates')
def get_plate_data():
    plate_dir = 'violations/plates'
    image_files = [f for f in os.listdir(plate_dir) if f.endswith('.jpg')]
    plate_texts = []
    if os.path.exists(f'{plate_dir}/plates.txt'):
        with open(f'{plate_dir}/plates.txt') as f:
            plate_texts = [line.strip() for line in f.readlines()]
    return jsonify({'images': image_files, 'texts': plate_texts})

@app.route('/plates/<filename>')
def get_plate_image(filename):
    return send_from_directory('violations/plates', filename)

def get_plate_text(plate_filename):
    """Extract plate text from the filename if it follows a specific pattern."""
    try:
        return os.path.splitext(plate_filename)[0].split('_')[-1]
    except:
        return None

def get_related_plates(violation_filename):
    """Find all plate images related to a violation."""
    base_name = os.path.splitext(violation_filename)[0]
    plate_dir = os.path.join('violations', 'plates')
    related_plates = []
    
    if not os.path.exists(plate_dir):
        return related_plates
        
    plate_prefix = f"plate_{base_name.split('_', 1)[1]}"
    
    for plate_file in os.listdir(plate_dir):
        if plate_file.startswith(plate_prefix) and plate_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            related_plates.append({
                'filename': plate_file,
                'text': get_plate_text(plate_file)
            })
    
    return related_plates

@app.route('/violations')
def view_violations():
    violations_dir = 'violations'
    violation_files = []

    plates_dir = os.path.join(violations_dir, 'plates')
    os.makedirs(plates_dir, exist_ok=True)

    for file in os.listdir(violations_dir):
        file_path = os.path.join(violations_dir, file)
        if file.lower().endswith(('.jpg', '.jpeg', '.png')) and os.path.isfile(file_path):
            timestamp = os.path.getmtime(file_path)
            timestamp_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

            plates = get_related_plates(file)
            plate_text = plates[0]['text'] if plates else None

            fname_lower = file.lower()
            if "redlight" in fname_lower or "red_light" in fname_lower or "red-light" in fname_lower:
                violation_type = "Red Light Violation"
            elif "pedestrian" in fname_lower:
                violation_type = "Pedestrian Crossing Violation"
            else:
                violation_type = "Other Violation"

            violation_files.append({
                'filename': file,
                'timestamp': timestamp_str,
                'plates': plates,
                'plate_text': plate_text,
                'violation_type': violation_type
            })

    violation_files.sort(key=lambda x: x['timestamp'], reverse=True)

    return render_template_string(HTML_TEMPLATE, violations=violation_files)

@app.route('/violations/<filename>')
def get_violation_image(filename):
    return send_from_directory('violations', filename)

if __name__ == '__main__':
    os.makedirs('violations', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
