import cv2
import mediapipe as mp
import math
import gradio as gr
import numpy as np
import threading
import time
from deepface import DeepFace
import speech_recognition as sr
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter, defaultdict
import joblib
import re
burnout_theme = gr.Theme(
    # Set the primary and secondary colors to match your design
    primary_hue="pink",
    secondary_hue="gray"
).set(
    # General colors and backgrounds
    body_background_fill="#f7f3e8",
    body_background_fill_dark="#f7f3e8",
    block_background_fill="#fff",
    block_background_fill_dark="#fff",
    block_info_text_color="#333",
    block_info_text_color_dark="#333",
    block_label_text_color="#333",
    block_label_text_color_dark="#333",
    block_title_text_color="#333",
    block_title_text_color_dark="#333",
    
    # Text colors
    body_text_color="#333",
    body_text_color_dark="#333",
    body_text_color_subdued="#666",
    
    # Accent colors for buttons, tabs, and titles
    color_accent_soft="#e5989b",
    color_accent_soft_dark="#d4888b",
)

# Configuration
RUN_DURATION = 20

# Define a custom Gradio theme

EMOTION_BUCKETS = {
    'positive': ['happy'],
    'neutral': ['neutral'],
    'negative': ['angry', 'sad', 'fear', 'disgust', 'surprise']
}

# Detailed emotions with feature-based subtypes
DETAILED_EMOTIONS = {
    'angry': {
        'subtypes': {
            'annoyance': {'brow_furrow': 1, 'eye_squint': 1},
            'frustration': {'brow_furrow': 2, 'jaw_clench': 1, 'lip_press': 1},
            'rage': {'mouth_open': 2, 'brow_furrow': 3, 'eye_squint': 2},
        },
        'description': "Eyebrows lowered, eyes glaring, mouth tense"
    },
    'happy': {
        'subtypes': {
            'amusement': {'mouth_smile': 1},
            'joy': {'mouth_smile': 2, 'eye_crinkle': 1},
            'elation': {'mouth_smile': 3, 'eye_crinkle': 2},
        },
        'description': "Cheeks raised, mouth corners up"
    },
    'sad': {
        'subtypes': {
            'disappointment': {'mouth_frown': 1},
            'sorrow': {'mouth_frown': 2, 'brow_raise': 1},
            'grief': {'mouth_frown': 3, 'brow_raise': 2},
        },
        'description': "Inner eyebrows raised, mouth corners down"
    },
    'fear': {
        'subtypes': {
            'anxiety': {'eye_widen': 1},
            'alarm': {'eye_widen': 2, 'mouth_open': 1},
            'terror': {'eye_widen': 3, 'mouth_open': 2},
        },
        'description': "Eyes wide, eyebrows raised, mouth open"
    },
    'disgust': {
        'subtypes': {
            'revulsion': {'nose_wrinkle': 2, 'upper_lip_raise': 2},
            'contempt': {'lip_curl': 1}
        },
        'description': "Nose wrinkled, upper lip raised"
    },
    'surprise': {
        'subtypes': {
            'suddenness': {'eye_widen': 1, 'mouth_open': 1},
            'astonishment': {'eye_widen': 2, 'jaw_drop': 2},
            'awe': {'eye_widen': 2, 'brow_raise': 2},
            'warning': {'brow_furrow': 1, 'eye_squint': 1, 'mouth_open': 1}
        },
        'description': "Eyebrows raised, jaw drop"
    },
    'neutral': {
        'subtypes': {
            'calm': {},
            'contemplative': {'brow_furrow': 0.5},
        },
        'description': "Relaxed facial muscles"
    }
}


sentiment_analyzer = SentimentIntensityAnalyzer()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
INNER_MOUTH = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 95, 88, 178, 87, 14, 317, 402, 318, 324]

questions = [
    "Do you often feel exhausted or stressed after classes and hostel work, even with enough sleep?",
    "Do you feel like you're just going through the motions, not really enjoying hostel or college life?",
    "Do you avoid people or group work because you just don’t feel like talking?",
    "Do you feel proud of the progress you're making in your academics or personal goals?",
    "Do you feel that your efforts are leading to meaningful outcomes or recognition?",
]
options = [
    ["Never", "Rarely", "Sometimes", "Often", "Always"],
    ["Never", "Rarely", "Sometimes", "Often", "Always"],
    ["Never", "Rarely", "Sometimes", "Often", "Always"],
    ["Never", "Rarely", "Sometimes", "Often", "Always"],
    ["Never", "Rarely", "Sometimes", "Often", "Always"],
]

# ---------------------------
# Emotion Detector (unchanged logic, but safe to call from a thread)
# ---------------------------
class CompleteEmotionDetector:
    def __init__(self):
        self.bucket_counts = defaultdict(int)
        self.detailed_emotion_counts = defaultdict(int)
        self.subtype_counts = defaultdict(lambda: defaultdict(int))
        self.lock = threading.Lock()
        # ADD THESE TWO LINES:
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        

    def _get_emotion_bucket(self, emotion):
        for bucket, emotions_list in EMOTION_BUCKETS.items():
            if emotion in emotions_list:
                return bucket
        return "unknown"

    def detect_anger_features(self, roi_gray, w, h):
        features = {'brow_furrow': 0, 'eye_squint': 0, 'jaw_clench': 0, 'lip_press': 0, 'mouth_open': 0}
        brow_region = roi_gray[h//5:h//2, w//3:2*w//3]
        if brow_region.size > 0:
            sobel_x = cv2.Sobel(brow_region, cv2.CV_64F, 1, 0, ksize=3)
            if np.mean(np.abs(sobel_x)) > 18: features['brow_furrow'] = 2
        eyes = self.eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
        if len(eyes) > 0:
            avg_eye_height = np.mean([eh for (ex, ey, ew, eh) in eyes])
            if (avg_eye_height / h) < 0.1: features['eye_squint'] = 2
        mouth_roi = roi_gray[2*h//3:h, w//4:3*w//4]
        if mouth_roi.size > 0:
            thresh = cv2.adaptiveThreshold(mouth_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                area = cv2.contourArea(max(contours, key=cv2.contourArea))
                normalized_area = area / (mouth_roi.shape[0] * mouth_roi.shape[1])
                if normalized_area > 0.15: features['mouth_open'] = 2
                elif normalized_area < 0.02: features['lip_press'] = 2
        return features

    def detect_happy_features(self, roi_gray, w, h):
        features = {'mouth_smile': 0, 'eye_crinkle': 0}
        mouth_roi = roi_gray[2*h//3:h, w//5:4*w//5]
        if mouth_roi.size > 0:
            thresh = cv2.adaptiveThreshold(mouth_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x_c, y_c, w_c, h_c = cv2.boundingRect(largest_contour)
                if w_c > 0 and h_c > 0 and (w_c / h_c) > 2.0:
                    features['mouth_smile'] = 2
        eyes = self.eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
        if len(eyes) > 0:
            eye_crinkle_variance = 0
            for (ex, ey, ew, eh) in eyes:
                corner_roi = roi_gray[ey:ey+eh, max(0, ex-ew//2):ex]
                if corner_roi.size > 0:
                    eye_crinkle_variance += cv2.Laplacian(corner_roi, cv2.CV_64F).var()
            if eye_crinkle_variance / len(eyes) > 100:
                features['eye_crinkle'] = 2
        return features

    def detect_sad_features(self, roi_gray, w, h):
        features = {'mouth_frown': 0, 'brow_raise': 0}
        mouth_roi = roi_gray[2*h//3:h, w//4:3*w//4]
        if mouth_roi.size > 0:
            thresh = cv2.adaptiveThreshold(mouth_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                epsilon = 0.02 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                if len(approx) >= 6:
                    mouth_y_center = h * 0.75
                    lower_points = [p[0][1] for p in approx if p[0][1] > mouth_y_center]
                    upper_points = [p[0][1] for p in approx if p[0][1] < mouth_y_center]
                    if lower_points and upper_points and np.mean(lower_points) > np.mean(upper_points) + (h * 0.05):
                        features['mouth_frown'] = 2
        brow_center_roi = roi_gray[h//5:h//2, w//3:2*w//3]
        if brow_center_roi.size > 0 and cv2.Laplacian(brow_center_roi, cv2.CV_64F).var() > 75:
            features['brow_raise'] = 1
        return features

    def detect_fear_features(self, roi_gray, w, h):
        features = {'eye_widen': 0, 'mouth_open': 0}
        eyes = self.eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
        if len(eyes) > 0:
            aspect_ratios = []
            for (ex, ey, ew, eh) in eyes:
                if ew > 0:
                    aspect_ratios.append(eh / ew)
            if aspect_ratios:
                avg_ratio = np.mean(aspect_ratios)
                if avg_ratio > 0.8:
                    features['eye_widen'] = 2
        mouth_roi = roi_gray[2*h//3:h, w//4:3*w//4]
        if mouth_roi.size > 0:
            thresh = cv2.adaptiveThreshold(mouth_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                x_c, y_c, w_c, h_c = cv2.boundingRect(cnt)
                if h_c > 0 and (w_c / h_c) < 1.5:
                    features['mouth_open'] = 2
        return features

    def detect_disgust_features(self, roi_gray, w, h):
        features = {'nose_wrinkle': 0, 'upper_lip_raise': 0, 'lip_curl': 0}
        nose_region = roi_gray[h//4:h//2, w//3:2*w//3]
        if nose_region.size > 0 and cv2.Laplacian(nose_region, cv2.CV_64F).var() > 70:
            features['nose_wrinkle'] = 2
        mouth_roi = roi_gray[2*h//3:h, w//4:3*w//4]
        if mouth_roi.size > 0:
            thresh = cv2.adaptiveThreshold(mouth_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                if len(cnt) > 0 and cnt[:, 0, 1].min() < mouth_roi.shape[0] * 0.2:
                    features['upper_lip_raise'] = 1
        return features

    def detect_surprise_features(self, roi_gray, w, h):
        features = {'eye_widen': 0, 'mouth_open': 0, 'jaw_drop': 0, 'brow_raise': 0}
        eyes = self.eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
        if len(eyes) > 0:
            aspect_ratios = []
            for (ex, ey, ew, eh) in eyes:
                if ew > 0:
                    aspect_ratios.append(eh / ew)
            if aspect_ratios:
                avg_ratio = np.mean(aspect_ratios)
                if avg_ratio > 0.9:
                    features['eye_widen'] = 3
        mouth_roi = roi_gray[2*h//3:h, w//4:3*w//4]
        if mouth_roi.size > 0:
            thresh = cv2.adaptiveThreshold(mouth_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                x_c, y_c, w_c, h_c = cv2.boundingRect(cnt)
                if h_c > 0:
                    normalized_area = cv2.contourArea(cnt) / (mouth_roi.shape[0] * mouth_roi.shape[1])
                    if normalized_area > 0.18 and (w_c / h_c) < 1.0:
                        features['mouth_open'] = 3
                        features['jaw_drop'] = 2
        brow_upper_region = roi_gray[h//10:h//3, w//4:3*w//4]
        if brow_upper_region.size > 0 and cv2.Laplacian(brow_upper_region, cv2.CV_64F).var() > 90:
            features['brow_raise'] = 2
        return features

    def determine_subtype(self, emotion, features):
        scores = defaultdict(int)
        if emotion not in DETAILED_EMOTIONS or 'subtypes' not in DETAILED_EMOTIONS[emotion]:
            return ""
        for subtype, feature_map in DETAILED_EMOTIONS[emotion]['subtypes'].items():
            score = sum(features.get(feature, 0) * weight for feature, weight in feature_map.items())
            scores[subtype] = score
        if not any(scores.values()):
            return list(DETAILED_EMOTIONS[emotion]['subtypes'].keys())[0] if DETAILED_EMOTIONS[emotion]['subtypes'] else ""
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def analyze_frame(self, frame):
        """
        This method is designed to be called in a background thread. It calls DeepFace
        (which is CPU/GPU heavy) and updates internal counters under a lock.
        """
        try:
            results = DeepFace.analyze(
                frame, actions=['emotion'],
                detector_backend='mtcnn',
                enforce_detection=False, silent=True
            )
            with self.lock:
                if results:
                    result = results[0]
                    region = result.get('region', {})
                    x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)
                    # validate region and proceed like before
                    if w > 0 and h > 0 and x >= 0 and y >= 0 and (x + w) <= frame.shape[1] and (y + h) <= frame.shape[0]:
                        dominant_emotion_deepface = result.get('dominant_emotion', 'neutral')
                        roi_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                        feature_detectors = {
                            'angry': self.detect_anger_features, 'happy': self.detect_happy_features,
                            'sad': self.detect_sad_features, 'fear': self.detect_fear_features,
                            'disgust': self.detect_disgust_features, 'surprise': self.detect_surprise_features,
                        }
                        current_subtype = ""
                        if dominant_emotion_deepface in feature_detectors:
                            features = feature_detectors[dominant_emotion_deepface](roi_gray, w, h)
                            current_subtype = self.determine_subtype(dominant_emotion_deepface, features)
                            if current_subtype:
                                self.subtype_counts[dominant_emotion_deepface][current_subtype] += 1
                        elif dominant_emotion_deepface == 'neutral':
                            # record neutral default subtype
                            self.subtype_counts['neutral']['calm'] += 1
                        self.detailed_emotion_counts[dominant_emotion_deepface] += 1
                        bucket = self._get_emotion_bucket(dominant_emotion_deepface)
                        self.bucket_counts[bucket] += 1
        except Exception:
            # keep silent on failures to avoid crashing UI
            pass

    def get_stats(self):
        with self.lock:
            total = sum(self.bucket_counts.values())
            pos = (self.bucket_counts['positive'] / total) * 100 if total else 0.0
            neu = (self.bucket_counts['neutral'] / total) * 100 if total else 0.0
            neg = (self.bucket_counts['negative'] / total) * 100 if total else 0.0
            return (pos, neu, neg)

# ---------------------------
# Face Analyzer (restructured for per-frame processing)
# ---------------------------
# --- Face Analyzer Class ---
class FaceAnalyzer:
    def __init__(self, emotion_detector_instance):
        # Initialize lock FIRST
        self.face_lock = threading.Lock()
        self.emotion_detector = emotion_detector_instance
        self.cap = None
        self.running = False
        # Now call reset
        self.reset()

    def reset(self):
        with self.face_lock:
            self.ear_values = []
            self.mar_values = []
            self.start_time = None
            self.running = False

    def calculate_ear(self, eye_points, landmarks):
        """Calculate Eye Aspect Ratio"""
        try:
            # Vertical distances
            v1 = math.dist(landmarks[eye_points[1]], landmarks[eye_points[5]])
            v2 = math.dist(landmarks[eye_points[2]], landmarks[eye_points[4]])
            # Horizontal distance
            h = math.dist(landmarks[eye_points[0]], landmarks[eye_points[3]])
            return (v1 + v2) / (2.0 * h) if h > 0 else 0.0
        except:
            return 0.0

    def calculate_mar(self, landmarks):
        """Calculate Mouth Aspect Ratio using inner mouth points"""
        try:
            points = [landmarks[i] for i in INNER_MOUTH]
            if len(points) < 2:
                return 0.0
            
            # Get mouth height (vertical distance)
            y_coords = [p[1] for p in points]
            mouth_height = max(y_coords) - min(y_coords)
            
            # Get mouth width using outer points (approximate)
            if len(landmarks) > 60:
                mouth_width = math.dist(landmarks[61], landmarks[291])  # Left to right mouth corner
            else:
                mouth_width = math.dist(points[0], points[6]) if len(points) > 6 else 1.0
            
            return mouth_height / mouth_width if mouth_width > 0 else 0.0
        except:
            return 0.0

    def start_run(self):
        with self.face_lock:
            self.ear_values = []
            self.mar_values = []
            self.running = True
            self.start_time = time.time()
        
        # Reset emotion detector counts for new session
        with self.emotion_detector.lock:
            self.emotion_detector.bucket_counts = defaultdict(int)
            self.emotion_detector.detailed_emotion_counts = defaultdict(int)
            self.emotion_detector.subtype_counts = defaultdict(lambda: defaultdict(int))

    def process_frame(self, frame):
        """Process a single frame for EAR, MAR, and emotion analysis"""
        if not self.running:
            return frame, "---", "---", "Not Running"

        elapsed = time.time() - self.start_time
        if elapsed >= RUN_DURATION:
            self.running = False
            return frame, "---", "---", "Stopped (Time reached)"

        # Convert to RGB for mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        ear, mar = None, None
        
        if results.multi_face_landmarks:
            try:
                face_landmarks = results.multi_face_landmarks[0]
                landmarks = [(int(p.x * frame.shape[1]), int(p.y * frame.shape[0])) 
                            for p in face_landmarks.landmark]

                # Calculate EAR and MAR
                left_ear = self.calculate_ear(LEFT_EYE, landmarks)
                right_ear = self.calculate_ear(RIGHT_EYE, landmarks)
                ear = (left_ear + right_ear) / 2.0
                mar = self.calculate_mar(landmarks)

                with self.face_lock:
                    if ear is not None:
                        self.ear_values.append(ear)
                    if mar is not None:
                        self.mar_values.append(mar)

                # Draw landmarks for visualization
                for idx in LEFT_EYE + RIGHT_EYE + INNER_MOUTH:
                    if idx < len(landmarks):
                        frame = frame.copy()
                        cv2.circle(frame, tuple(map(int, landmarks[idx])), 1, (0, 255, 0), -1)

                # Display metrics on frame
                cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"MAR: {mar:.2f}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            except Exception as e:
                print(f"Landmark processing error: {e}")
                ear, mar = 0.3, 0.4  # Default values if processing fails

        # Run emotion detection in background
        try:
            if not hasattr(self, 'emotion_thread') or not self.emotion_thread.is_alive():
                frame_copy = frame.copy()
                self.emotion_thread = threading.Thread(
                    target=self.emotion_detector.analyze_frame, 
                    args=(frame_copy,),
                    daemon=True
                )
                self.emotion_thread.start()
        except Exception as e:
            print(f"Emotion thread error: {e}")

        # Add timer overlay
        remaining_time = max(0, RUN_DURATION - int(time.time() - self.start_time))
        timer_text = f"Time left: {remaining_time}s"
        text_size = cv2.getTextSize(timer_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = frame.shape[1] - text_size[0] - 20
        text_y = text_size[1] + 20
        
        # Create semi-transparent background for timer
        overlay = frame.copy()
        cv2.rectangle(overlay, (text_x - 10, text_y - text_size[1] - 10),
                     (text_x + text_size[0] + 10, text_y + 10), (0, 0, 0), -1)
        alpha = 0.4
        processed_frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        cv2.putText(processed_frame, timer_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        return (cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB),
               f"{ear:.2f}" if ear is not None else "---",
               f"{mar:.2f}" if mar is not None else "---",
               f"Running ({elapsed:.1f}s)")

    def finalize_stats(self):
        """Compute final statistics after analysis completes"""
        with self.face_lock:
            if not self.ear_values or not self.mar_values:
                return (0, 0, 0, 0), (0, 0, 0)

            avg_ear = np.mean(self.ear_values)
            std_ear = np.std(self.ear_values)
            avg_mar = np.mean(self.mar_values)
            std_mar = np.std(self.mar_values)

        # Get emotion statistics from the detector
        emotion_stats = self.emotion_detector.get_stats()

        return (avg_ear, std_ear, avg_mar, std_mar), emotion_stats

    def stop(self):
        """Stop the analysis and clean up resources"""
        self.running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if hasattr(self, 'emotion_thread') and self.emotion_thread.is_alive():
            self.emotion_thread.join(timeout=1.0)


# --- Initialize global objects ---
emotion_detector = CompleteEmotionDetector()
face_analyzer = FaceAnalyzer(emotion_detector)  # Pass the emotion detector
stored_data = {}

# Voice Analysis Functions
def analyze_voice(audio):

    max_secs = 10
    if isinstance(audio, tuple) and len(audio) == 2:
        sample_rate, audio_data = audio
        max_samples = int(sample_rate * max_secs)
        if len(audio_data.shape) == 2:  # stereo
            audio_data = audio_data[:max_samples, :]
        else:
            audio_data = audio_data[:max_samples]
    import tempfile
    import scipy.io.wavfile as wav
    import speech_recognition as sr
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import numpy as np

    sentiment_analyzer = SentimentIntensityAnalyzer()

    if audio is None:
        return "No audio input.", "N/A", "N/A"

    audio_path = None
    try:
        if isinstance(audio, tuple) and len(audio) == 2:
            sample_rate, audio_data = audio
            if np.issubdtype(audio_data.dtype, np.floating):
                if np.max(np.abs(audio_data)) <= 1.01:
                    audio_data = (audio_data * 32767).astype(np.int16)
                else:
                    audio_data = audio_data.astype(np.int16)
            temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            wav.write(temp.name, sample_rate, audio_data)
            audio_path = temp.name
        elif isinstance(audio, str):
            audio_path = audio
        else:
            return "Unrecognized audio input", "N/A", "N/A"

        r = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_sr = r.record(source)
        try:
            text = r.recognize_google(audio_sr)
        except (sr.UnknownValueError, sr.RequestError):
            text = ""
        sentiment = sentiment_analyzer.polarity_scores(text) if text else {"pos": 0, "neu": 1, "neg": 0, "compound": 0}
        sentiment_str = (
            f"Pos: {sentiment['pos']:.2f}, Neu: {sentiment['neu']:.2f}, "
            f"Neg: {sentiment['neg']:.2f}, Comp: {sentiment['compound']:.2f}"
        )
        return text, sentiment_str, "N/A"  # Return only the 3 expected values
    except Exception as e:
        return f"Error: {str(e)}", "N/A", "N/A"

def aggregate_results(questionnaire_answers, face_stats, emotion_stats, voice_stats, subtype_counts):
    print("face_stats:", face_stats)
    print("emotion_stats:", emotion_stats)
    print("voice_stats:", voice_stats)
    avg_ear, std_ear, avg_mar, std_mar = face_stats
    pos_perc, neu_perc, neg_perc = emotion_stats
    text, sentiment_scores, pitch = voice_stats
    subtype_table = []
    for emotion, subtypes in subtype_counts.items():
        for subtype, count in subtypes.items():
            subtype_table.append([emotion, subtype, count])
    result_dict = {
        "Questionnaire Answers": questionnaire_answers,
        "EAR": f"{avg_ear:.2f} ± {std_ear:.2f}",
        "MAR": f"{avg_mar:.2f} ± {std_mar:.2f}",
        "Emotions": {
            "Positive %": f"{pos_perc:.1f}%",
            "Neutral %": f"{neu_perc:.1f}%",
            "Negative %": f"{neg_perc:.1f}%"
        },
        "Voice Transcript": text,
        "Sentiment": sentiment_scores,
        "Pitch": "N/A",
        "Subtypes": subtype_table
    }
    return result_dict

answer_mapping = {
    "Never": 0,
    "Rarely": 1,
    "Sometimes": 2,
    "Often": 3,
    "Always": 4,
    "I enjoy my work. I have no symptoms of burnout": 0,
    "Occasionally I am under stress, and I don’t always have as much energy as I once did, but I don’t feel burned out": 1,
    "I am definitely burning out and have one or more symptoms of burnout, such as physical and emotional exhaustion": 2,
    "The symptoms of burnout that I’m experiencing won’t go away. I think about frustration at work a lot": 3,
    "I feel completely burned out and often wonder if I can go on. I am at the point where I may need some changes or may need to seek some sort of help": 4,
}
scaler=joblib.load('scaler.joblib')
model = joblib.load('model.joblib')
def preprocess_input(raw_dict, scaler):
    # Map questionnaire answers to ints
    q_encoded = [answer_mapping.get(ans, -1) for ans in raw_dict["Questionnaire Answers"]]
    if -1 in q_encoded:
        raise ValueError("Unknown answer detected in questionnaire answers.")

    # Parse EAR and MAR (only average, ignore std)
    EAR_avg = float(raw_dict["EAR"].split(" ± ")[0])
    MAR_avg = float(raw_dict["MAR"].split(" ± ")[0])

    # Parse emotions percentages (remove % and convert)
    emotions = raw_dict["Emotions"]
    pos_emotion = float(emotions["Positive %"].rstrip('%'))
    neu_emotion = float(emotions["Neutral %"].rstrip('%'))
    neg_emotion = float(emotions["Negative %"].rstrip('%'))

    # Parse sentiment scores (extract floats from string)
    sentiment_str = raw_dict["Sentiment"]
    sentiment_vals = re.findall(r'[-+]?\d*\.\d+|\d+', sentiment_str)
    # Expected order: Pos, Neu, Neg, Comp
    sentiment_vals = list(map(float, sentiment_vals))
    if len(sentiment_vals) != 4:
        raise ValueError("Sentiment string format unexpected.")
    sent_pos, sent_neu, sent_neg, sent_comp = sentiment_vals

    # Build input feature vector as per model's expected order:
    features = q_encoded + [
        EAR_avg, 0,  # Avg_EAR, placeholder for Std_EAR (not provided)
        MAR_avg, 0,  # Avg_MAR, placeholder for Std_MAR (not provided)
        pos_emotion, neu_emotion, neg_emotion,
        sent_pos, sent_neu, sent_neg, sent_comp
    ]

    # Convert to numpy array 2D for scaler and model
    X = np.array(features).reshape(1, -1)

    # Apply scaler
    X_scaled = scaler.transform(X)

    # Optionally convert to pandas DataFrame with feature names if needed
    # feature_names = [...] # from model.feature_names_
    # X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

    return X_scaled

def predict_burnout(raw_dict):
    X_processed = preprocess_input(raw_dict, scaler)
    pred = model.predict(X_processed)
    # Assuming regression or float prediction; if classifier, adapt accordingly
    output_score = float(pred[0])
    return output_score
import gradio as gr
import os

custom_css = """
    /* Main body and font styling */
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700&family=Playfair+Display:wght@400;500;600;700&display=swap');

    body, html, .gradio-app {
        background-color: #f7f3e8 !important; /* Soft beige background */
        font-family: 'Montserrat', sans-serif;
        color: #333 !important;
    }
    
    /* Specific overrides for Hugging Face's dark mode to ensure text is visible */
    [data-theme='dark'] body, 
    [data-theme='dark'] .gradio-app,
    [data-theme='dark'] .gradio-container,
    [data-theme='dark'] .nav-bar,
    [data-theme='dark'] .hero-section,
    [data-theme='dark'] .functionality-section,
    [data-theme='dark'] .technical-details-section,
    [data-theme='dark'] .gr-text-box,
    [data-theme='dark'] .gr-markdown,
    [data-theme='dark'] .gr-radio-label {
        background-color: #f7f3e8 !important;
        color: #333 !important;
    }
    
    /* Override input and label colors for better visibility in dark mode */
    [data-theme='dark'] .gr-input, 
    [data-theme='dark'] .gr-label,
    [data-theme='dark'] .gr-output-text,
    [data-theme='dark'] .gr-json-preview {
        background-color: #fff !important;
        color: #333 !important;
    }
    
    [data-theme='dark'] .gradio-tabs .tab-button {
      background-color: #e5989b !important;
      color: #fff !important;
    }
    
    [data-theme='dark'] .gradio-tabs .tab-button.selected {
      background-color: #d4888b !important;
    }
    
    [data-theme='dark'] .gradio-tabs .tab-button.selected::before {
      background-color: #d4888b !important;
    }
    
    [data-theme='dark'] .gradio-tabs .tab-button:hover {
      background-color: #d4888b !important;
    }

    .gradio-container {
        max-width: 1600px !important;
        margin: 0 auto;
        padding: 20px;
        background-color: #f7f3e8 !important;
        box-shadow: none !important;
    }
    
    /* Navigation Bar Styling */
    .nav-bar {
        background-color: #fff !important;
        padding: 20px;
        border-radius: 30px;
        display: flex;
        align-items: center;
        gap: 40px;
    }
    .nav-left {
        display: flex;
        align-items: center;
    }
    .nav-right {
        display: flex;
        align-items: center;
        gap: 20px;
        margin-left: auto;
        text-align: right;
    }
    .nav-links span {
        font-weight: 500;
        cursor: pointer;
        padding: 5px 10px;
        border-radius: 5px;
        transition: background-color 0.2s;
    }
    .nav-links span:hover {
        background-color: rgba(255, 255, 255, 0.5);
    }
    .logo {
        font-family: 'Playfair Display', serif;
        font-size: 1.5em;
        font-weight: 800;
        color: #e5989b !important; /* Soft pink accent */
    }
    .search-icon {
        cursor: pointer;
    }
    .theme-text {
      color: #777 !important;
      font-size: 0.8em;
      font-style: italic;
    }

    /* Hero Section Styling */
    .hero-section {
        background-color: #fff !important;
        padding: 60px;
        margin-top: 20px;
        border-radius: 30px;
        display: flex;
        align-items: center;
        gap: 40px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
    }
    .hero-text-container {
        flex: 1;
        padding-right: 20px;
    }
    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 3em;
        font-weight: 700;
        line-height: 1.4;
        color: #e5989b !important; /* Soft pink accent */
    }
    .hero-subtitle {
        color: #666 !important;
        line-height: 1.6;
        margin-top: 15px;
    }
    .learn-more-btn {
        background-color: #e5989b !important; /* Soft pink accent */
        color: #fff !important;
        padding: 12px 25px;
        border-radius: 25px;
        margin-top: 30px;
        font-weight: 600;
        transition: background-color 0.2s;
    }
    .learn-more-btn:hover {
        background-color: #d4888b !important; /* Slightly darker pink for hover */
    }
    .hero-image-container {
        flex: 1;
        text-align: right;
    }
    .hero-image {
        max-width: 75%;
        border-radius: 30px;
    }

    /* Functionality Section Styling */
    .functionality-section {
        background-color: #fff !important;
        padding: 40px;
        margin-top: 40px;
        border-radius: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
        max-width: 1600px;
        margin-left: auto;
        margin-right: auto;
    }
    .functionality-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.5em;
        font-weight: 600;
        text-align: center;
        margin-bottom: 20px;
        color: #e5989b !important; /* Soft pink accent */
    }
    .gradio-tabs {
        border-radius: 15px !important;
    }
    .gradio-tabs .tab-button {
      border: 1px solid #e5989b !important; /* Soft pink accent */
      color: #e5989b !important;
      background-color: #f7f3e8 !important;
    }
    .gradio-tabs .tab-button.selected {
      color: #fff !important;
      background-color: #e5989b !important;
    }
    .gradio-tabs .tab-button.selected::before {
      background-color: #e5989b !important;
    }


    /* Scrollbar for Questionnaire Tab */
    #tab-0 .scrollable {
        max-height: 400px;
        overflow-y: auto;
    }

    /* Technical Details Section Styling */
    .technical-details-section {
        background-color: #fff !important;
        padding: 60px;
        border-radius: 30px;
        display: flex;
        align-items: center;
        gap: 40px;
        margin-top: 40px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
    }
    .expertise-text-container {
        flex: 1;
    }
    .expertise-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.5em;
        font-weight: 600;
        color: #e5989b !important; /* Soft pink accent */
        margin-bottom: 10px;
    }
    .expertise-subtitle {
        color: #666 !important;
        line-height: 1.6;
        margin-bottom: 30px;
    }
    .expertise-image {
        max-width: 100%;
        border-radius: 30px;
    }
    .feature-list {
        display: flex;
        flex-direction: column;
        gap: 20px;
    }
    .feature-item {
        display: block;
    }
    .feature-icon {
        display: none;
    }
    .feature-text {
        padding-left: 0;
    }
    .feature-text h4 {
        margin: 0;
        font-weight: 600;
        color: #4a4a4a !important;
    }
    .feature-text p {
        margin: 5px 0 0;
        color: #666 !important;
    }
    
    /* Removed metrics section CSS */
"""

with gr.Blocks(theme=burnout_theme, css=custom_css) as demo:
    # This is the JavaScript from your previous prompt to force the light theme.
    demo.load(
        None,
        None,
        js="""
        () => {
            const params = new URLSearchParams(window.location.search);
            if (!params.has('__theme')) {
                params.set('__theme', 'light');
                window.location.search = params.toString();
            }
        }
        """,
    )

    # Mimicking a navigation bar
    with gr.Row(elem_classes=["nav-bar"]):
        with gr.Column(elem_classes=["nav-left"], scale=1):
            gr.Markdown('<div class="logo">Burnout Detection System</div>', elem_classes="logo")
        with gr.Column(elem_classes=["nav-right"], scale=2):
            gr.Markdown('<span class="theme-text">best viewed in light mode</span>', elem_classes="theme-text")

    # Hero Section
    with gr.Row(elem_classes=["hero-section"]):
        with gr.Column(elem_classes=["hero-text-container"]):
            gr.Markdown('<h1 class="hero-title">A New Standard in Burnout Detection</h1>', elem_classes="hero-title")
            gr.Markdown('<p class="hero-subtitle">Move beyond simple, subjective questionnaires and embrace a new era of proactive well-being management. Our innovative system provides a truly comprehensive and objective analysis of your mental and emotional state by harnessing the unparalleled power of multimodal AI. It intelligently integrates real-time analysis of subtle facial micro-expressions, nuanced vocal sentiment patterns, and a carefully designed clinical questionnaire to detect the earliest and often unseen signs of burnout with unprecedented accuracy and reliability. Specifically engineered for high-achieving students and dedicated professionals navigating immense pressure and demanding environments, this cutting-edge tool delivers deeply personalized insights and actionable feedback you can genuinely trust. Empower yourself to take definitive control of your mental health with a system that profoundly understands the unseen strain and silent battles you face. Begin to proactively manage your stress levels, build resilience, and effectively prevent burnout before it escalates into an overwhelming and debilitating challenge, ensuring your sustained well-being and peak performance.</p>')
        with gr.Column(elem_classes=["hero-image-container"]):
            gr.Image("hero.png", elem_classes="hero-image")

    # Functionality Section
    with gr.Column(elem_classes=["functionality-section"], elem_id="functionality-section"):
        
        gr.Markdown('<h2 class="functionality-title">Try it out!</h2>')
        with gr.Tabs(elem_classes=["gradio-tabs"]) as tabs:
            # Questionnaire
            with gr.TabItem("Questionnaire", id=0):
                with gr.Column(elem_classes=["scrollable"]):
                    radio_components = []
                    for i, (q, opts) in enumerate(zip(questions, options)):
                        with gr.Row():
                            gr.Markdown(f"**{i+1}. {q}**")
                            r = gr.Radio(opts, label="Your answer")
                            radio_components.append(r)
                submit_btn = gr.Button("Submit Answers")

            # Face Analysis
            with gr.TabItem("Face Analysis", id=1):
                with gr.Column():
                    webcam = gr.Image(label="Webcam Feed", sources="webcam", streaming=True)
                    with gr.Row():
                        ear_output = gr.Textbox(label="Avg EAR ± Std", value="---")
                        mar_output = gr.Textbox(label="Avg MAR ± Std", value="---")
                    with gr.Row():
                        pos_emotion_output = gr.Textbox(label="Positive Emotion %", value="---")
                        neu_emotion_output = gr.Textbox(label="Neutral Emotion %", value="---")
                        neg_emotion_output = gr.Textbox(label="Negative Emotion %", value="---")
                    status_output = gr.Textbox(label="Status", value="Ready")
                    start_face_btn = gr.Button("Start Face Analysis")
                    update_stats_btn = gr.Button("Update Stats")
                    next_to_voice_btn = gr.Button("Proceed to Voice Analysis", interactive=False)

            # Voice Analysis
            with gr.TabItem("Voice Analysis", id=2):
                audio_in = gr.Audio(label="Your Recording", sources="microphone", streaming=False)
                record_btn = gr.Button("Analyze Voice")
                text_output = gr.Textbox(label="Transcribed Text")
                sentiment_output = gr.Textbox(label="Sentiment Analysis (Pos, Neu, Neg, Comp)")
                pitch_output = gr.Textbox(label="Pitch (Hz) ± Std Dev")
                proceed_to_results_btn = gr.Button("Proceed to Results")

            # Results
            with gr.TabItem("Results", id=3):
                results_json = gr.JSON(label="Full Results (Questionnaire, EAR/MAR, Emotion, Voice, Subtypes)")
                burnout_score_output = gr.Textbox(label="Burnout Score", interactive=False)

    # Technical Details Section (How it Works)
    with gr.Row(elem_classes=["technical-details-section"]):
        with gr.Column(elem_classes=["expertise-text-container"]):
            gr.Markdown('<h2 class="expertise-title">How It Works</h2>')
            gr.Markdown('<p class="expertise-subtitle">Our system employs a sophisticated blend of techniques to deliver accurate results by analyzing multiple data points.</p>')
            with gr.Column(elem_classes=["feature-list"]):
                with gr.Row(elem_classes=["feature-item"]):
                    gr.Markdown("<h4>Video Analysis</h4>")
                    gr.Markdown("<p>Utilizes advanced facial recognition and computer vision to analyze eye and mouth movements, along with subtle emotional cues.</p>")
                with gr.Row(elem_classes=["feature-item"]):
                    gr.Markdown("<h4>Audio Analysis</h4>")
                    gr.Markdown("<p>Analyzes vocal tone, pitch, and sentiment to detect emotional cues and stress levels in your voice.</p>")
                with gr.Row(elem_classes=["feature-item"]):
                    gr.Markdown("<h4>Questionnaire</h4>")
                    gr.Markdown("<p>A standardized self-assessment questionnaire provides crucial self-reported data to complement the sensor-based analysis.</p>")
                gr.Button("Learn More", elem_classes="learn-more-btn")
        with gr.Column(elem_classes=["expertise-image-container"]):
            gr.Image("https://placehold.co/400x550/e5989b/fff?text=How+It+Works+Image", elem_classes="expertise-image")

    # --- Event Handlers ---

    # Questionnaire submit → store answers + go to Face Analysis tab
    def questionnaire_to_face(*answers):
        stored_data["questionnaire"] = list(answers)
        return gr.Tabs(selected=1)

    submit_btn.click(
        fn=questionnaire_to_face,
        inputs=radio_components,
        outputs=tabs,
    )

    # Start face analysis
    def start_face_analysis():
        face_analyzer.start_run()
        return "Started", gr.update(interactive=False)

    start_face_btn.click(fn=start_face_analysis, outputs=[status_output, next_to_voice_btn])

    # Webcam streaming
    def stream_frame(frame):
        return face_analyzer.process_frame(frame)

    webcam.stream(
        fn=stream_frame,
        inputs=[webcam],
        outputs=[webcam, ear_output, mar_output, status_output],
    )

    # Update stats after face analysis
    def update_stats_button():
        face_stats, emotion_stats = face_analyzer.finalize_stats()
        stored_data["face"] = face_stats
        stored_data["emotion"] = emotion_stats
        stored_data["subtypes"] = dict(emotion_detector.subtype_counts)
        avg_ear, std_ear, avg_mar, std_mar = face_stats
        pos_perc, neu_perc, neg_perc = emotion_stats
        avg_ear_s = f"{avg_ear:.2f} ± {std_ear:.2f}"
        avg_mar_s = f"{avg_mar:.2f} ± {std_mar:.2f}"
        return (
            avg_ear_s,
            avg_mar_s,
            "Finished Face Analysis",
            f"{pos_perc:.1f}%",
            f"{neu_perc:.1f}%",
            f"{neg_perc:.1f}%",
            gr.update(interactive=True),
        )

    update_stats_btn.click(
        fn=update_stats_button,
        outputs=[ear_output, mar_output, status_output,
                 pos_emotion_output, neu_emotion_output, neg_emotion_output,
                 next_to_voice_btn],
    )

    # Proceed to Voice Analysis tab
    def go_to_voice():
        #face_analyzer.stop()
        return gr.Tabs(selected=2)

    next_to_voice_btn.click(fn=go_to_voice, outputs=[tabs])

    # Voice analysis
    def voice_analysis_and_store(audio):
        text, sentiment_str, pitch_display = analyze_voice(audio)
        stored_data["voice"] = (text, sentiment_str, pitch_display)
        return text, sentiment_str, pitch_display

    record_btn.click(
        fn=voice_analysis_and_store,
        inputs=[audio_in],
        outputs=[text_output, sentiment_output, pitch_output],
    )

    # Proceed to Results
    def proceed_to_results():
        required_keys = ["questionnaire", "face", "emotion", "voice", "subtypes"]
        missing_keys = [key for key in required_keys if key not in stored_data]

        if missing_keys:
            return {"Error": f"Missing data: {', '.join(missing_keys)}. Please complete all sections."}, "", gr.Tabs(selected=2)

        try:
            # Defensive cleaning as you had
            subtype_counts = stored_data.get("subtypes", {})
            cleaned_subtypes = {str(emotion): {str(k): int(v) for k, v in subs.items()} 
                                for emotion, subs in subtype_counts.items()}
            
            results = aggregate_results(
                stored_data["questionnaire"],
                stored_data["face"],
                stored_data["emotion"],
                stored_data["voice"],
                cleaned_subtypes,
            )
            # Store results dict to later use for model prediction
            stored_data["results"] = results

            # Get burnout score from model prediction
            burnout_score = predict_burnout(results)  # Your integration function
            
            return results, f"{burnout_score:.2f}", gr.Tabs(selected=3)
        except Exception as e:
            return {"Error": f"Failed to generate results: {str(e)}"}, "", gr.Tabs(selected=2)



    proceed_to_results_btn.click(
        fn=proceed_to_results,
        outputs=[results_json,burnout_score_output, tabs],
    )

    
if __name__ == "__main__":
    demo.launch(share=True)
