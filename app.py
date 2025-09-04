import cv2
import mediapipe as mp
import math
import gradio as gr
import numpy as np
import threading
import time
from deepface import DeepFace
import sounddevice as sd
import scipy.io.wavfile as wav
import speech_recognition as sr
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import os
import datetime
from collections import defaultdict
import librosa

# --- Google Sheets Imports ---
import gspread
from google.oauth2.service_account import Credentials
import json # New import

# --- Configuration for Emotion Detection ---
RUN_DURATION = 20

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

# --- Analysis Class Definitions ---

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
    "Overall, based on your definition of burnout, how would you rate your level of burnout?"
]
options = [
    ["Never", "Rarely", "Sometimes", "Often", "Always"],
    ["Never", "Rarely", "Sometimes", "Often", "Always"],
    ["Never", "Rarely", "Sometimes", "Often", "Always"],
    ["Never", "Rarely", "Sometimes", "Often", "Always"],
    ["Never", "Rarely", "Sometimes", "Often", "Always"],
    ["I enjoy my work. I have no symptoms of burnout", "Occasionally I am under stress, and I don’t always have as much energy as I once did, but I don’t feel burned out", "I am definitely burning out and have one or more symptoms of burnout, such as physical and emotional exhaustion", "The symptoms of burnout that I’m experiencing won’t go away. I think about frustration at work a lot", "I feel completely burned out and often wonder if I can go on. I am at the point where I may need some changes or may need to seek some sort of help"]
]

# --- Complete Emotion Detector Class (with custom features) ---
class CompleteEmotionDetector:
    def __init__(self):
        self.bucket_counts = defaultdict(int)
        self.detailed_emotion_counts = defaultdict(int)
        self.subtype_counts = defaultdict(lambda: defaultdict(int)) # Store subtype counts
        self.total_frames = 0
        self.lock = threading.Lock()
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
        try:
            results = DeepFace.analyze(
                frame, actions=['emotion'],
                detector_backend='mtcnn',
                enforce_detection=False, silent=True
            )

            with self.lock:
                if results:
                    result = results[0]
                    region = result['region']
                    x, y, w, h = region['x'], region['y'], region['w'], region['h']

                    if w > 0 and h > 0 and x >= 0 and y >= 0 and (x + w) <= frame.shape[1] and (y + h) <= frame.shape[0]:
                        dominant_emotion_deepface = result['dominant_emotion']

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
                        elif dominant_emotion_deepface == 'neutral' and 'subtypes' in DETAILED_EMOTIONS['neutral']:
                            current_subtype = 'calm' # Default neutral subtype
                            self.subtype_counts[dominant_emotion_deepface][current_subtype] += 1

                        self.detailed_emotion_counts[dominant_emotion_deepface] += 1
                        dominant_emotion_bucket = self._get_emotion_bucket(dominant_emotion_deepface)
                        self.bucket_counts[dominant_emotion_bucket] += 1

        except Exception as e:
            pass

    def get_stats(self):
        with self.lock:
            total_emotion_frames = sum(self.bucket_counts.values())
            pos_perc = (self.bucket_counts['positive'] / total_emotion_frames) * 100 if total_emotion_frames > 0 else 0.0
            neu_perc = (self.bucket_counts['neutral'] / total_emotion_frames) * 100 if total_emotion_frames > 0 else 0.0
            neg_perc = (self.bucket_counts['negative'] / total_emotion_frames) * 100 if total_emotion_frames > 0 else 0.0
            return (pos_perc, neu_perc, neg_perc)

# --- Corrected and Optimized FaceAnalyzer Class ---
class FaceAnalyzer:
    def __init__(self, emotion_detector_instance):
        self.running = False
        self.cap = None
        self.ear_values = []
        self.mar_values = []
        self.face_lock = threading.Lock()
        self.emotion_detector = emotion_detector_instance
        self.emotion_thread = None
        self.last_emotion_analysis_time = 0

    def calculate_ear(self, eye_points, landmarks):
        v1 = math.dist(landmarks[eye_points[1]], landmarks[eye_points[5]])
        v2 = math.dist(landmarks[eye_points[2]], landmarks[eye_points[4]])
        h = math.dist(landmarks[eye_points[0]], landmarks[eye_points[3]])
        return (v1 + v2) / (2.0 * h) if h > 0 else 0.0

    def calculate_inner_mar(self, landmarks):
        points = [landmarks[i] for i in INNER_MOUTH]
        y_coords = [p[1] for p in points]
        mouth_height = max(y_coords) - min(y_coords)
        cheek_width = math.dist(landmarks[454], landmarks[234])
        return mouth_height / cheek_width if cheek_width > 0 else 0.0

    def stream_frames(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not self.cap.isOpened():
            yield (None, "Error: Webcam not found.", "---", "Error", None, None, gr.Button(interactive=False), gr.Image(visible=False))
            return

        self.running = True
        with self.face_lock:
            self.ear_values = []
            self.mar_values = []

        with self.emotion_detector.lock:
            self.emotion_detector.bucket_counts = defaultdict(int)
            self.emotion_detector.detailed_emotion_counts = defaultdict(int)
            self.emotion_detector.subtype_counts = defaultdict(lambda: defaultdict(int))


        start_time = time.time()

        while self.running and self.cap.isOpened() and (time.time() - start_time) < RUN_DURATION:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            ear, mar = None, None
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                landmarks = [(int(p.x * frame.shape[1]), int(p.y * frame.shape[0])) for p in face_landmarks.landmark]

                left_ear = self.calculate_ear(LEFT_EYE, landmarks)
                right_ear = self.calculate_ear(RIGHT_EYE, landmarks)
                ear = (left_ear + right_ear) / 2.0
                mar = self.calculate_inner_mar(landmarks)

                with self.face_lock:
                    self.ear_values.append(ear)
                    self.mar_values.append(mar)

                for idx in LEFT_EYE + RIGHT_EYE + INNER_MOUTH:
                    cv2.circle(frame, landmarks[idx], 1, (0, 255, 0), -1)

                cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"MAR: {mar:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Asynchronous Emotion Analysis
            if not self.emotion_thread or not self.emotion_thread.is_alive():
                frame_copy = frame.copy()
                self.emotion_thread = threading.Thread(target=self.emotion_detector.analyze_frame, args=(frame_copy,))
                self.emotion_thread.start()

            remaining_time = max(0, RUN_DURATION - int(time.time() - start_time))
            timer_text = f"Time left: {remaining_time}s"
            text_size, _ = cv2.getTextSize(timer_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            text_x = frame.shape[1] - text_size[0] - 20
            text_y = text_size[1] + 20
            overlay = frame.copy()
            cv2.rectangle(overlay, (text_x - 10, text_y - text_size[1] - 10),
                          (text_x + text_size[0] + 10, text_y + 10), (0, 0, 0), -1)
            alpha = 0.4
            processed_frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            cv2.putText(processed_frame, timer_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            yield (cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB),
                   f"{ear:.2f}" if ear is not None else "---",
                   f"{mar:.2f}" if mar is not None else "---",
                   "Analyzing...",
                   None, None, gr.Button(interactive=False), gr.Image(visible=True))

        self.stop()

        with self.face_lock:
            avg_ear = np.mean(self.ear_values) if self.ear_values else 0
            std_ear = np.std(self.ear_values) if self.ear_values else 0
            avg_mar = np.mean(self.mar_values) if self.mar_values else 0
            std_mar = np.std(self.mar_values) if self.mar_values else 0

        face_stats_tuple = (avg_ear, std_ear, avg_mar, std_mar)
        emotion_stats_tuple = self.emotion_detector.get_stats()

        yield (None,
               f"{avg_ear:.2f} ± {std_ear:.2f}",
               f"{avg_mar:.2f} ± {std_mar:.2f}",
               "Finished Face Analysis",
               face_stats_tuple,
               emotion_stats_tuple,
               gr.Button(interactive=True),
               gr.Image(visible=False))

    def stop(self):
        self.running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None
        if self.emotion_thread and self.emotion_thread.is_alive():
            self.emotion_thread.join()

# Voice Analysis Functions
def analyze_voice(duration=10):
    sample_rate = 44100
    gr.Info("Speak Now!")

    audio_file = "recording.wav"

    try:
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()

        audio_data = audio.flatten()

        wav.write(audio_file, sample_rate, (audio_data * 32767).astype(np.int16))

        r = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio_sr = r.record(source)

        try:
            text = r.recognize_google(audio_sr)
        except (sr.UnknownValueError, sr.RequestError):
            text = ""

        sentiment = sentiment_analyzer.polarity_scores(text) if text else {"pos": 0, "neu": 1, "neg": 0, "compound": 0}

        try:
            y, sr_librosa = librosa.load(audio_file, sr=sample_rate)
            f0, _, _ = librosa.template.logfreq(y, sr=sr_librosa, fmin=50, fmax=500)
            pitch_values_librosa = f0[~np.isnan(f0)]

            if len(pitch_values_librosa) > 0:
                mean_pitch = np.mean(pitch_values_librosa)
                std_pitch = np.std(pitch_values_librosa)
            else:
                mean_pitch = None
                std_pitch = None
        except Exception as e:
            print(f"Error during librosa pitch calculation: {e}")
            mean_pitch = None
            std_pitch = None

        return (
            (sample_rate, audio_data),
            text,
            f"Pos: {sentiment['pos']:.2f}, Neu: {sentiment['neu']:.2f}, Neg: {sentiment['neg']:.2f}, Comp: {sentiment['compound']:.2f}",
            f"{mean_pitch:.2f} Hz ± {std_pitch:.2f} Hz" if mean_pitch is not None else "No pitch detected",
            text, sentiment, mean_pitch, std_pitch
        )

    except Exception as e:
        return (
            (44100, np.zeros(1)),
            f"Error: {str(e)}",
            "N/A",
            "N/A",
            "", {"pos": 0, "neu": 1, "neg": 0, "compound": 0}, None, None
        )

# --- Data Storage Functions ---

def save_to_google_sheet(questionnaire_answers, face_stats, emotion_stats, voice_stats, subtype_counts):
    # --- IMPORTANT: Configure these two variables ---
    # We now read credentials from a local file, which is fine for local dev
    SERVICE_ACCOUNT_FILE = 'service_account_key.json'
    SPREADSHEET_NAME = 'data'
    # -----------------------------------------------

    scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']

    try:
        creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=scopes)
        client = gspread.authorize(creds)

        spreadsheet = client.open(SPREADSHEET_NAME)
        worksheet = spreadsheet.sheet1

        avg_ear, std_ear, avg_mar, std_mar = face_stats
        pos_perc, neu_perc, neg_perc = emotion_stats
        _transcribed_text, sentiment_scores, mean_pitch, std_pitch = voice_stats

        # Flatten subtype counts for the row - dynamically generate columns based on actual subtypes detected
        all_possible_subtype_keys = []
        for emotion_type, details in DETAILED_EMOTIONS.items():
            for subtype_name in details['subtypes'].keys():
                all_possible_subtype_keys.append(f"{emotion_type.capitalize()}_{subtype_name.replace('_', ' ').title().replace(' ', '')}_Count")

        question_cols = [f"Q{i+1}" for i in range(len(questionnaire_answers))]
        face_cols = ["Avg_EAR", "Std_EAR", "Avg_MAR", "Std_MAR"]
        emotion_cols = ["Positive_Percent", "Neutral_Percent", "Negative_Percent"]
        voice_cols = ["Sentiment_Pos", "Sentiment_Neu", "Sentiment_Neg", "Sentiment_Comp", "Mean_Pitch", "Std_Pitch"]

        # Ensure header is present (only on the first run)
        if not worksheet.get_all_values():
            header = ["Timestamp"] + question_cols + face_cols + emotion_cols + voice_cols + all_possible_subtype_keys
            worksheet.append_row(header)

        # Prepare the data row to append
        data_row = [
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ] + questionnaire_answers + [
            avg_ear, std_ear, avg_mar, std_mar,
            pos_perc, neu_perc, neg_perc,
            sentiment_scores['pos'], sentiment_scores['neu'], sentiment_scores['neg'], sentiment_scores['compound'],
            mean_pitch if mean_pitch is not None else "N/A", std_pitch if std_pitch is not None else "N/A"
        ]

        # Append subtype counts for the current row, matching the header order
        current_subtype_values = []
        for emotion_type, details in DETAILED_EMOTIONS.items():
            for subtype_name in details['subtypes'].keys():
                # Get count, default to 0 if not found for this specific run
                count = subtype_counts.get(emotion_type, {}).get(subtype_name, 0)
                current_subtype_values.append(count)

        data_row.extend(current_subtype_values)

        worksheet.append_row(data_row)

        return "Analysis complete. Data saved to Google Sheet."

    except gspread.exceptions.SpreadsheetNotFound:
        return f"Error: The specified Google Sheet was not found. Please check the name ('{SPREADSHEET_NAME}') and sharing settings."
    except Exception as e:
        return f"An error occurred while saving to Google Sheet: {e}"


# Instantiate the analysis classes
emotion_detector = CompleteEmotionDetector()
face_analyzer = FaceAnalyzer(emotion_detector)
stored_data = {}

with gr.Blocks() as demo:
    with gr.Tabs(selected=0) as tabs:
        with gr.Tab("Questionnaire", id=0):
            radio_components = []
            with gr.Column():
                gr.Markdown("## Please answer these 5 questions")
                for i, (question, opts) in enumerate(zip(questions, options)):
                    with gr.Row():
                        gr.Markdown(f"**{i+1}. {question}**")
                        radio = gr.Radio(opts, label="Your answer")
                        radio_components.append(radio)
                submit_btn = gr.Button("Submit Answers")

        with gr.Tab("Face Analysis", id=1):
            with gr.Column():
                gr.Markdown("## Face Analysis (20 seconds)")
                gr.Markdown("Please look at the camera for the duration of the recording. Your EAR and MAR will be displayed.")
                webcam = gr.Image(label="Webcam Feed")
                with gr.Row():
                    ear_output = gr.Textbox(label="Avg EAR ± Std", value="---")
                    mar_output = gr.Textbox(label="Avg MAR ± Std", value="---")
                with gr.Row():
                    pos_emotion_output = gr.Textbox(label="Positive Emotion %", value="---")
                    neu_emotion_output = gr.Textbox(label="Neutral Emotion %", value="---")
                    neg_emotion_output = gr.Textbox(label="Negative Emotion %", value="---")
                status_output = gr.Textbox(label="Status", value="Ready")

                face_stats_state = gr.State()
                emotion_stats_state = gr.State()
                subtype_counts_state = gr.State() # New state for subtype data

                start_face_btn = gr.Button("Start Face Analysis")
                next_to_voice_btn = gr.Button("Proceed to Voice Analysis", interactive=False)

        with gr.Tab("Voice Analysis", id=2):
            with gr.Column():
                gr.Markdown("## Voice Analysis (10 seconds)")
                gr.Markdown("Click the button and speak clearly for 10 seconds.")
                record_btn = gr.Button("Record and Analyze Voice")
                audio_out = gr.Audio(label="Your Recording")
                text_output = gr.Textbox(label="Transcribed Text")
                sentiment_output = gr.Textbox(label="Sentiment Analysis (Pos, Neu, Neg, Comp)")
                pitch_output = gr.Textbox(label="Pitch (Hz) ± Std Dev")
                save_data_btn = gr.Button("Save All Data")

    final_status = gr.Textbox(label="Final Status", value="")

    def questionnaire_to_face(*answers):
        stored_data["questionnaire"] = list(answers)
        return gr.Tabs(selected=1)

    submit_btn.click(
        fn=questionnaire_to_face,
        inputs=radio_components,
        outputs=tabs
    )

    def handle_face_analysis_completion(face_stats_tuple, emotion_stats_tuple):
        stored_data["face"] = face_stats_tuple
        stored_data["emotion"] = emotion_stats_tuple
        # Storing subtype counts from the emotion detector state
        stored_data["subtypes"] = dict(emotion_detector.subtype_counts)

        avg_ear, std_ear, avg_mar, std_mar = face_stats_tuple
        pos_perc, neu_perc, neg_perc = emotion_stats_tuple

        return (f"{avg_ear:.2f} ± {std_ear:.2f}",
                f"{avg_mar:.2f} ± {std_mar:.2f}",
                "Finished Face Analysis",
                f"{pos_perc:.1f}%",
                f"{neu_perc:.1f}%",
                f"{neg_perc:.1f}%",
                gr.Button(interactive=True),
                gr.Image(visible=False))

    start_face_btn.click(
        fn=face_analyzer.stream_frames,
        outputs=[webcam, ear_output, mar_output, status_output, face_stats_state, emotion_stats_state, next_to_voice_btn, webcam]
    ).then(
        fn=handle_face_analysis_completion,
        inputs=[face_stats_state, emotion_stats_state],
        outputs=[ear_output, mar_output, status_output, pos_emotion_output, neu_emotion_output, neg_emotion_output, next_to_voice_btn, webcam]
    )

    def transition_to_voice(face_stats, emotion_stats):
        face_analyzer.stop()
        if face_stats and emotion_stats:
            return gr.Tabs(selected=2)
        else:
            return gr.Tabs(selected=1)

    next_to_voice_btn.click(
        fn=transition_to_voice,
        inputs=[face_stats_state, emotion_stats_state],
        outputs=[tabs]
    )

    def voice_analysis_and_store():
        voice_results = analyze_voice()
        stored_data["voice"] = voice_results[-4:]
        return voice_results[:-4]

    record_btn.click(
        fn=voice_analysis_and_store,
        outputs=[audio_out, text_output, sentiment_output, pitch_output]
    )

    def save_all_data_to_gsheet():
        if "questionnaire" in stored_data and "face" in stored_data and "emotion" in stored_data and "voice" in stored_data and "subtypes" in stored_data:
            return save_to_google_sheet(
                stored_data["questionnaire"],
                stored_data["face"],
                stored_data["emotion"],
                stored_data["voice"],
                stored_data["subtypes"]
            )
        else:
            return "Incomplete data. Please complete all sections."

    save_data_btn.click(
        fn=save_all_data_to_gsheet, # Changed to use the Google Sheet saving function
        outputs=final_status
    )


if __name__ == "__main__":
    demo.launch()
    #server_name="0.0.0.0", server_port=10000
