# import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disables those warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Reduces TensorFlow verbosity


# import cv2
# from deepface import DeepFace
# import matplotlib.pyplot as plt
# import time

# # List of emotions we want to detect
# emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral', 
#                  'contempt', 'calm', 'confused', 'bored']

# def process_video(video_path, output_path=None, frame_skip=5):
#     """
#     Process a video file to detect emotions in each frame
    
#     Args:
#         video_path (str): Path to the input video file
#         output_path (str, optional): Path to save the output video. If None, won't save
#         frame_skip (int): Process every nth frame to improve performance
#     """
#     # Open the video file
#     cap = cv2.VideoCapture('vaibhav.mp4')
    
#     if not cap.isOpened():
#         print("Error: Could not open video file.")
#         return
    
#     # Get video properties for output if needed
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
    
#     # Initialize video writer if output path is provided
#     if output_path:
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
#     frame_count = 0
#     processed_frames = 0
    
#     while cap.isOpened():
#         ret, frame = cap.read()
        
#         if not ret:
#             break
            
#         frame_count += 1
        
#         # Skip frames to improve processing speed
#         if frame_count % frame_skip != 0:
#             continue
            
#         try:
#             # Convert frame to RGB (DeepFace uses RGB)
#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
#             # Analyze the frame for emotions
#             results = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False)
            
#             # Process each face found in the frame
#             for result in results:
#                 emotions = result['emotion']
#                 region = result['region']
                
#                 # Get the dominant emotion
#                 dominant_emotion = max(emotions, key=emotions.get)
#                 emotion_score = emotions[dominant_emotion]
                
#                 # Draw rectangle around the face
#                 x, y, w, h = region['x'], region['y'], region['w'], region['h']
#                 cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
#                 # Display emotion text
#                 text = f"{dominant_emotion}: {emotion_score:.1f}%"
#                 cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
#                             0.8, (0, 255, 0), 2, cv2.LINE_AA)
                
#                 # Print results to console
#                 print(f"Frame {frame_count}: {text}")
                
#         except Exception as e:
#             print(f"Error processing frame {frame_count}: {str(e)}")
#             continue
            
#         processed_frames += 1
        
#         # Display the processed frame
#         cv2.imshow('Emotion Detection', frame)
        
#         # Write the frame to output video if needed
#         if output_path:
#             out.write(frame)
            
#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
            
#     # Release resources
#     cap.release()
#     if output_path:
#         out.release()
#     cv2.destroyAllWindows()
    
#     print(f"Processing complete. Processed {processed_frames} frames out of {frame_count}.")

# if __name__ == "__main__":
#     # Example usage
#     input_video = "input_video.mp4"  # Replace with your video path
#     output_video = "output_video.mp4"  # Set to None if you don't want to save
    
#     print("Starting emotion detection...")
#     start_time = time.time()
    
#     process_video(input_video, output_video, frame_skip=5)
    
#     end_time = time.time()
#     print(f"Total processing time: {end_time - start_time:.2f} seconds")

######################################################################################################################################

# import cv2
# from deepface import DeepFace
# from collections import defaultdict

# # Emotion configuration
# EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
# MIN_CONFIDENCE = 15  # Minimum percentage to count an emotion

# # Frame counters
# emotion_frame_counts = defaultdict(int)
# total_frames = 0

# def analyze_frame(frame):
#     global total_frames
#     total_frames += 1
    
#     try:
#         # Analyze the frame
#         results = DeepFace.analyze(
#             cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
#             actions=['emotion'],
#             detector_backend='mtcnn',
#             enforce_detection=False,
#             silent=True
#         )
        
#         if results and 'emotion' in results[0]:
#             emotions = results[0]['emotion']
#             region = results[0]['region']
            
#             # Update frame counts for emotions above threshold
#             for emotion, score in emotions.items():
#                 if score >= MIN_CONFIDENCE:
#                     emotion_frame_counts[emotion] += 1
            
#             # Draw face rectangle and emotion info
#             x, y, w, h = region['x'], region['y'], region['w'], region['h']
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
#             # Display emotions
#             y_offset = y - 40
#             for emotion in EMOTIONS:
#                 if emotion in emotions and emotions[emotion] >= MIN_CONFIDENCE:
#                     text = f"{emotion}: {emotions[emotion]:.1f}%"
#                     cv2.putText(frame, text, (x, y_offset), 
#                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
#                                (0, 255, 255), 1)
#                     y_offset -= 25
                    
#     except Exception as e:
#         pass
        
#     return frame

# def print_emotion_stats():
#     print("\n=== Emotion Detection Statistics ===")
#     print(f"Total frames processed: {total_frames}")
#     print("\nFrames per emotion:")
#     for emotion in EMOTIONS:
#         count = emotion_frame_counts.get(emotion, 0)
#         print(f"{emotion.capitalize()}: {count} frames")

# def main():
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Error: Could not open webcam.")
#         return
    
#     # Set webcam resolution
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
#     print("Starting webcam emotion detection...")
#     print("Press 'q' to quit and view statistics")
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
            
#         # Mirror the frame
#         frame = cv2.flip(frame, 1)
        
#         # Analyze and display
#         frame = analyze_frame(frame)
#         cv2.imshow('Emotion Detection', frame)
        
#         # Exit on 'q' key
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
            
#     # Release resources and print stats
#     cap.release()
#     cv2.destroyAllWindows()
#     print_emotion_stats()

# if __name__ == "__main__":
#     main()

#####################################################################################################################################

# import cv2
# import numpy as np
# from deepface import DeepFace
# from collections import defaultdict

# # FACS-based emotion configuration
# EMOTION_CONFIG = {
#     'happy': {
#         'required_aus': [6, 12],  # Cheek raiser + lip corner puller
#         'optional_aus': [],
#         'description': "Smile with eye wrinkles"
#     },
#     'sad': {
#         'required_aus': [1, 4, 15],  # Inner brow raiser + brow lowerer + lip corner depressor
#         'optional_aus': [17],  # Chin raiser
#         'description': "Downturned mouth with inner brow raise"
#     },
#     'surprise': {
#         'required_aus': [1, 2, 5, 26],  # Brow raiser + upper lid raiser + jaw drop
#         'optional_aus': [],
#         'description': "Raised brows with wide eyes"
#     },
#     'fear': {
#         'required_aus': [1, 2, 4, 5, 20],  # Brow raiser + lid raiser + lip stretcher
#         'optional_aus': [25, 26],  # Lips part + jaw drop
#         'description': "Tensed brows with stretched lips"
#     },
#     'disgust': {
#         'required_aus': [9, 4],  # Nose wrinkler + brow lowerer
#         'optional_aus': [10, 15],  # Upper lip raiser + lip corner depressor
#         'description': "Wrinkled nose with lowered brows"
#     },
#     'angry': {
#         'required_aus': [4, 5, 7, 23],  # Brow lowerer + lid tightener + lips tightener
#         'optional_aus': [9, 10],  # Nose wrinkler + upper lip raiser
#         'description': "Lowered brows with tightened eyes"
#     },
#     'neutral': {
#         'required_aus': [],
#         'optional_aus': [],
#         'description': "Relaxed facial muscles"
#     }
# }

# MIN_CONFIDENCE = 20  # Minimum confidence percentage
# FRAME_HISTORY = 5    # Number of frames to consider for smoothing

# class EmotionDetector:
#     def __init__(self):
#         self.emotion_counts = defaultdict(int)
#         self.total_frames = 0
#         self.history = []
        
#     def detect_facs_features(self, frame, face_region):
#         """Simulate FACS feature detection (in a real implementation, use a dedicated FACS detector)"""
#         x, y, w, h = face_region
#         roi = frame[y:y+h, x:x+w]
#         gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
#         # Placeholder for actual FACS detection - in practice you'd use:
#         # 1. Facial landmark detection (DLib, MediaPipe)
#         # 2. Geometric relationships between landmarks
#         # 3. Muscle activation patterns
        
#         # This is a simplified approximation:
#         features = {
#             'brow_raise': 0,  # AU1+2
#             'brow_lower': 0,   # AU4
#             'eye_widen': 0,    # AU5
#             'cheek_raise': 0,  # AU6
#             'lid_tighten': 0,  # AU7
#             'nose_wrinkle': 0, # AU9
#             'lip_corner_up': 0, # AU12
#             'lip_corner_down': 0, # AU15
#             'chin_raise': 0,    # AU17
#             'lip_stretch': 0,   # AU20
#             'jaw_drop': 0       # AU26
#         }
        
#         # Simple heuristics (replace with proper FACS analysis)
#         if gray.mean() > 100:  # Brighter areas might indicate raised features
#             features['cheek_raise'] = 1
#             features['lip_corner_up'] = 1
            
#         return features
    
#     def verify_emotion(self, emotion, features):
#         """Verify if detected AUs match expected FACS patterns"""
#         config = EMOTION_CONFIG[emotion]
        
#         # Check required AUs
#         for au in config['required_aus']:
#             au_name = {
#                 1: 'brow_raise', 2: 'brow_raise', 4: 'brow_lower',
#                 5: 'eye_widen', 6: 'cheek_raise', 7: 'lid_tighten',
#                 9: 'nose_wrinkle', 12: 'lip_corner_up', 15: 'lip_corner_down',
#                 17: 'chin_raise', 20: 'lip_stretch', 23: 'lip_tighten',
#                 26: 'jaw_drop'
#             }.get(au, '')
            
#             if not features.get(au_name, 0):
#                 return False
                
#         return True
    
#     def process_frame(self, frame):
#         self.total_frames += 1
        
#         try:
#             # Get DeepFace analysis
#             results = DeepFace.analyze(
#                 cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
#                 actions=['emotion'],
#                 detector_backend='mtcnn',
#                 enforce_detection=False,
#                 silent=True
#             )
            
#             if not results or 'emotion' not in results[0]:
#                 return frame
                
#             result = results[0]
#             emotions = result['emotion']
#             region = result['region']
#             x, y, w, h = region['x'], region['y'], region['w'], region['h']
            
#             # Get FACS features
#             facs_features = self.detect_facs_features(frame, (x, y, w, h))
            
#             # Verify and count emotions
#             for emotion, score in emotions.items():
#                 if score >= MIN_CONFIDENCE and self.verify_emotion(emotion, facs_features):
#                     self.emotion_counts[emotion] += 1
            
#             # Draw results
#             self.draw_analysis(frame, emotions, (x, y, w, h))
            
#         except Exception as e:
#             print(f"Processing error: {str(e)}")
            
#         return frame
    
#     def draw_analysis(self, frame, emotions, face_region):
#         x, y, w, h = face_region
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
#         # Display emotions
#         y_offset = y - 40
#         for emotion in EMOTION_CONFIG.keys():
#             if emotion in emotions and emotions[emotion] >= MIN_CONFIDENCE:
#                 count = self.emotion_counts.get(emotion, 0)
#                 text = f"{emotion}: {emotions[emotion]:.1f}% ({count} frames)"
#                 color = (0, 255, 0) if emotion == 'happy' else (0, 0, 255) if emotion == 'sad' else (0, 255, 255)
#                 cv2.putText(frame, text, (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
#                 y_offset -= 25
    
#     def print_stats(self):
#         print("\n=== FACS-Based Emotion Detection Results ===")
#         print(f"Total frames processed: {self.total_frames}")
#         print("\nEmotion breakdown:")
#         for emotion, config in EMOTION_CONFIG.items():
#             count = self.emotion_counts.get(emotion, 0)
#             percentage = (count / self.total_frames) * 100 if self.total_frames > 0 else 0
#             print(f"{emotion.capitalize():<8}: {count:>4} frames ({percentage:.1f}%) | {config['description']}")

# def main():
#     detector = EmotionDetector()
#     cap = cv2.VideoCapture(0)
    
#     if not cap.isOpened():
#         print("Error: Could not open webcam.")
#         return
    
#     # Set optimal resolution
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
#     print("Starting FACS-enhanced emotion detection...")
#     print("Press 'Q' to quit and view statistics")
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
            
#         frame = cv2.flip(frame, 1)
#         frame = detector.process_frame(frame)
        
#         cv2.imshow('FACS Emotion Detection', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
            
#     cap.release()
#     cv2.destroyAllWindows()
#     detector.print_stats()

# if __name__ == "__main__":
#     main()

#####################################################################################################################################

# import cv2
# import numpy as np
# from deepface import DeepFace
# from collections import defaultdict

# # Configuration
# EMOTION_CONFIG = {
#     'happy': {
#         'required_aus': [6, 12],  # Cheek raiser + lip corner puller
#         'description': "Smile with eye wrinkles",
#         'color': (0, 255, 0)  # Green
#     },
#     'sad': {
#         'required_aus': [1, 4, 15],  # Inner brow raiser + brow lowerer + lip corner depressor
#         'description': "Downturned mouth with inner brow raise",
#         'color': (0, 0, 255)  # Red
#     },
#     'crying': {
#         'required_features': ['tears', 'puffy_eyes', 'red_eyes', 'tense_eyebrows'],
#         'description': "Tears with sad facial features",
#         'color': (255, 0, 0)  # Blue
#     },
#     'surprise': {
#         'required_aus': [1, 2, 5, 26],  # Brow raiser + upper lid raiser + jaw drop
#         'description': "Raised brows with wide eyes",
#         'color': (0, 255, 255)  # Yellow
#     },
#     'fear': {
#         'required_aus': [1, 2, 4, 5, 20],  # Brow raiser + lid raiser + lip stretcher
#         'description': "Tensed brows with stretched lips",
#         'color': (255, 255, 0)  # Cyan
#     },
#     'disgust': {
#         'required_aus': [9, 4],  # Nose wrinkler + brow lowerer
#         'description': "Wrinkled nose with lowered brows",
#         'color': (0, 128, 128)  # Teal
#     },
#     'angry': {
#         'required_aus': [4, 5, 7, 23],  # Brow lowerer + lid tightener + lips tightener
#         'description': "Lowered brows with tightened eyes",
#         'color': (0, 0, 128)  # Maroon
#     },
#     'neutral': {
#         'required_aus': [],
#         'description': "Relaxed facial muscles",
#         'color': (255, 255, 255)  # White
#     }
# }

# MIN_CONFIDENCE = 25  # Minimum confidence percentage
# FRAME_SKIP = 2      # Process every 2nd frame for better performance

# class EmotionDetector:
#     def __init__(self):
#         self.emotion_counts = defaultdict(int)
#         self.total_frames = 0
#         self.processed_frames = 0
#         self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#         self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
#     def detect_crying_features(self, frame, face_region):
#         """Detect physical signs of crying using computer vision"""
#         x, y, w, h = face_region
#         roi = frame[y:y+h, x:x+w]
#         gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
#         crying_features = {
#             'tears': False,
#             'puffy_eyes': False,
#             'red_eyes': False,
#             'tense_eyebrows': False
#         }
        
#         # Detect eyes
#         eyes = self.eye_cascade.detectMultiScale(gray, 1.3, 5)
        
#         if len(eyes) >= 2:  # Both eyes detected
#             for (ex, ey, ew, eh) in eyes:
#                 eye_roi = roi[ey:ey+eh, ex:ex+ew]
                
#                 # Detect tears (bright spots)
#                 _, threshold = cv2.threshold(eye_roi, 220, 255, cv2.THRESH_BINARY)
#                 if np.sum(threshold) > 1000:
#                     crying_features['tears'] = True
                
#                 # Detect red eyes
#                 b, g, r = cv2.split(eye_roi)
#                 if np.mean(r) > np.mean(g) + 15 and np.mean(r) > np.mean(b) + 15:
#                     crying_features['red_eyes'] = True
                
#                 # Detect puffy eyes
#                 if eh > h/4.5:  # Larger than normal eye height
#                     crying_features['puffy_eyes'] = True
        
#         # Detect tense eyebrows (simplified)
#         eyebrow_region = gray[0:int(h/3), :]
#         if cv2.Laplacian(eyebrow_region, cv2.CV_64F).var() > 120:
#             crying_features['tense_eyebrows'] = True
            
#         return crying_features

#     def is_crying(self, crying_features):
#         """Determine if crying is occurring based on detected features"""
#         # Require tears plus at least two other signs
#         signs_present = sum(crying_features.values())
#         return crying_features['tears'] and signs_present >= 3

#     def process_frame(self, frame):
#         self.total_frames += 1
        
#         # Skip frames for better performance
#         if self.total_frames % FRAME_SKIP != 0:
#             return frame
            
#         self.processed_frames += 1
#         final_emotion = None
        
#         try:
#             # Get DeepFace analysis
#             results = DeepFace.analyze(
#                 cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
#                 actions=['emotion'],
#                 detector_backend='mtcnn',
#                 enforce_detection=False,
#                 silent=True
#             )
            
#             if not results or 'emotion' not in results[0]:
#                 return frame
                
#             result = results[0]
#             emotions = result['emotion']
#             region = result['region']
#             x, y, w, h = region['x'], region['y'], region['w'], region['h']
            
#             # Detect crying features
#             crying_features = self.detect_crying_features(frame, (x, y, w, h))
            
#             # Priority 1: Check for crying
#             if self.is_crying(crying_features):
#                 final_emotion = 'crying'
#             else:
#                 # Normal emotion detection
#                 max_emotion = max(emotions.items(), key=lambda x: x[1])
#                 if max_emotion[1] >= MIN_CONFIDENCE:
#                     final_emotion = max_emotion[0]
            
#             # Update counts
#             if final_emotion:
#                 self.emotion_counts[final_emotion] += 1
            
#             # Draw results
#             self.draw_analysis(frame, final_emotion, (x, y, w, h), crying_features)
            
#         except Exception as e:
#             print(f"Processing error: {str(e)}")
            
#         return frame

#     def draw_analysis(self, frame, emotion, face_region, crying_features):
#         x, y, w, h = face_region
#         color = EMOTION_CONFIG.get(emotion, {}).get('color', (255, 255, 255))
        
#         # Draw face rectangle
#         cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
#         # Display emotion and frame count
#         if emotion:
#             count = self.emotion_counts.get(emotion, 0)
#             text = f"{emotion}: {count} frames"
#             cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
#         # Display crying features if detected
#         if emotion == 'crying':
#             features_text = "Crying signs: " + ", ".join([f for f, v in crying_features.items() if v])
#             cv2.putText(frame, features_text, (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

#     def print_stats(self):
#         print("\n=== Enhanced Emotion Detection Results ===")
#         print(f"Total frames: {self.total_frames} (Processed: {self.processed_frames})")
#         print("\nEmotion breakdown:")
        
#         # Include all emotions plus crying
#         emotions_to_show = list(EMOTION_CONFIG.keys())
#         if 'crying' not in emotions_to_show:
#             emotions_to_show.append('crying')
        
#         for emotion in sorted(emotions_to_show):
#             count = self.emotion_counts.get(emotion, 0)
#             percentage = (count / self.processed_frames) * 100 if self.processed_frames > 0 else 0
#             desc = EMOTION_CONFIG.get(emotion, {}).get('description', '')
#             color = EMOTION_CONFIG.get(emotion, {}).get('color', (0, 0, 0))
#             color_name = f"RGB{color}"
            
#             print(f"{emotion.capitalize():<8}: {count:>4} frames ({percentage:.1f}%) | {color_name} | {desc}")

# def main():
#     detector = EmotionDetector()
#     cap = cv2.VideoCapture(0)
    
#     if not cap.isOpened():
#         print("Error: Could not open webcam.")
#         return
    
#     # Set optimal resolution
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
#     print("Starting enhanced emotion detection with crying recognition...")
#     print("Press 'Q' to quit and view statistics")
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
            
#         # Mirror the frame for more natural display
#         frame = cv2.flip(frame, 1)
        
#         # Process and display frame
#         frame = detector.process_frame(frame)
#         cv2.imshow('Emotion & Crying Detection', frame)
        
#         # Exit on 'q' key
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
            
#     # Release resources and print stats
#     cap.release()
#     cv2.destroyAllWindows()
#     detector.print_stats()

# if __name__ == "__main__":
#     main()

##################################################################################################################################

# import cv2
# import numpy as np
# from deepface import DeepFace
# from collections import defaultdict

# # Simplified configuration
# EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
# COLORS = {
#     'angry': (0, 0, 255),
#     'disgust': (0, 102, 51),
#     'fear': (255, 255, 0),
#     'happy': (0, 255, 0),
#     'sad': (255, 0, 0),
#     'surprise': (0, 255, 255),
#     'neutral': (255, 255, 255)
# }

# class SimpleEmotionDetector:
#     def __init__(self):
#         self.emotion_counts = defaultdict(int)
#         self.total_frames = 0
        
#     def process_frame(self, frame):
#         self.total_frames += 1
        
#         try:
#             # Simple face detection first
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, 1.3, 5)
            
#             for (x, y, w, h) in faces:
#                 # Get emotion analysis
#                 results = DeepFace.analyze(
#                     cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
#                     actions=['emotion'],
#                     detector_backend='ssd',
#                     enforce_detection=False,
#                     silent=True
#                 )
                
#                 if results and 'emotion' in results[0]:
#                     emotions = results[0]['emotion']
#                     dominant = max(emotions.items(), key=lambda x: x[1])
                    
#                     if dominant[1] > 20:  # Minimum confidence
#                         self.emotion_counts[dominant[0]] += 1
#                         self.draw_result(frame, dominant[0], (x, y, w, h))
        
#         except Exception as e:
#             print(f"Error: {str(e)}")
            
#         return frame
    
#     def draw_result(self, frame, emotion, face_region):
#         x, y, w, h = face_region
#         color = COLORS.get(emotion, (255, 255, 255))
        
#         # Draw face rectangle
#         cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        
#         # Draw emotion text with background
#         text = f"{emotion.upper()}"
#         (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
#         cv2.rectangle(frame, (x, y-40), (x+text_width, y), color, -1)
#         cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

# def main():
#     detector = SimpleEmotionDetector()
#     cap = cv2.VideoCapture(0)
    
#     # Verify webcam
#     if not cap.isOpened():
#         cap = cv2.VideoCapture(1)
#         if not cap.isOpened():
#             print("No webcam found!")
#             return
    
#     print("Press 'Q' to quit")
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
            
#         frame = cv2.flip(frame, 1)
#         frame = detector.process_frame(frame)
        
#         # Display instructions
#         cv2.putText(frame, "Show your face clearly", (10, 30), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
#         cv2.imshow('Emotion Detection', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
            
#     cap.release()
#     cv2.destroyAllWindows()
    
#     # Print results
#     print("\nDetection Results:")
#     for emotion, count in detector.emotion_counts.items():
#         print(f"{emotion}: {count} frames")

# if __name__ == "__main__":
#     main()

###################################################################################################################################

# import cv2
# import numpy as np
# from deepface import DeepFace
# from collections import defaultdict

# # Configuration
# EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
# COLORS = {
#     'angry': (0, 0, 255),
#     'disgust': (0, 153, 76),
#     'fear': (255, 255, 0),
#     'happy': (0, 255, 0),
#     'sad': (255, 0, 0),
#     'surprise': (0, 255, 255),
#     'neutral': (255, 255, 255)
# }

# class EmotionDashboard:
#     def __init__(self):
#         self.emotion_counts = defaultdict(int)
#         self.total_frames = 0
#         self.emotion_history = []
        
#     def update(self, frame):
#         self.total_frames += 1
        
#         try:
#             results = DeepFace.analyze(
#                 cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
#                 actions=['emotion'],
#                 detector_backend='ssd',
#                 enforce_detection=False,
#                 silent=True
#             )
            
#             if results and 'emotion' in results[0]:
#                 emotions = results[0]['emotion']
#                 region = results[0]['region']
                
#                 # Update counts and history
#                 dominant = max(emotions.items(), key=lambda x: x[1])
#                 if dominant[1] > 20:  # Minimum confidence
#                     self.emotion_counts[dominant[0]] += 1
#                     self.emotion_history.append(dominant[0])
                
#                 # Draw the dashboard
#                 self.draw_dashboard(frame, emotions, region)
                
#         except Exception as e:
#             print(f"Detection error: {str(e)}")
            
#         return frame
    
#     def draw_dashboard(self, frame, emotions, face_region):
#         x, y, w, h = face_region
        
#         # Draw face bounding box
#         dominant = max(emotions.items(), key=lambda x: x[1])
#         cv2.rectangle(frame, (x, y), (x+w, y+h), COLORS[dominant[0]], 2)
        
#         # Dashboard background
#         dashboard_height = 180
#         cv2.rectangle(frame, (0, 0), (300, dashboard_height), (40, 40, 40), -1)
        
#         # Header
#         cv2.putText(frame, "Emotion Dashboard", (10, 30), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
#         # Current emotion percentages
#         y_offset = 60
#         for emotion in EMOTIONS:
#             percentage = emotions.get(emotion, 0)
#             count = self.emotion_counts.get(emotion, 0)
            
#             # Format like "Happy: 96.4% (29 frames)"
#             text = f"{emotion.capitalize()}: {percentage:.1f}% ({count} frames)"
#             color = COLORS.get(emotion, (255, 255, 255))
            
#             cv2.putText(frame, text, (10, y_offset), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
#             y_offset += 25
        
#         # Dominant emotion summary
#         if self.emotion_history:
#             dominant = max(set(self.emotion_history), key=self.emotion_history.count)
#             cv2.putText(frame, f"Dominant: {dominant.upper()}", (10, y_offset+20),
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[dominant], 2)

# def main():
#     detector = EmotionDashboard()
#     cap = cv2.VideoCapture(0)
    
#     if not cap.isOpened():
#         print("Error: Could not open webcam.")
#         return
    
#     # Set resolution
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
#     print("Press 'Q' to quit")
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
            
#         frame = cv2.flip(frame, 1)
#         frame = detector.update(frame)
        
#         cv2.imshow('Emotion Dashboard', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
            
#     cap.release()
#     cv2.destroyAllWindows()
    
#     # Print final report
#     print("\nFinal Emotion Report:")
#     for emotion in EMOTIONS:
#         count = detector.emotion_counts.get(emotion, 0)
#         percentage = (count / detector.total_frames) * 100 if detector.total_frames > 0 else 0
#         print(f"{emotion.capitalize()}: {percentage:.1f}% ({count} frames)")

# if __name__ == "__main__":
#     main()

#################################################################################################################################

# import cv2
# import numpy as np
# from deepface import DeepFace

# # Configuration
# EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
# COLORS = {
#     'angry': (0, 0, 255),      # Red
#     'disgust': (0, 153, 0),    # Green
#     'fear': (255, 255, 0),     # Cyan
#     'happy': (0, 255, 0),      # Bright Green
#     'sad': (255, 0, 0),        # Blue
#     'surprise': (0, 255, 255), # Yellow
#     'neutral': (255, 255, 255) # White
# }

# def main():
#     # Initialize webcam
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("ERROR: Cannot open webcam. Trying alternate camera...")
#         cap = cv2.VideoCapture(1)
#         if not cap.isOpened():
#             print("FATAL: No webcam detected!")
#             return
    
#     # Set resolution
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
#     print("Press 'Q' to quit")
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("ERROR: Cannot read frame")
#             break
            
#         # Mirror the frame
#         frame = cv2.flip(frame, 1)
        
#         try:
#             # Try analysis with multiple backends
#             for backend in ['opencv', 'ssd', 'mtcnn']:
#                 try:
#                     results = DeepFace.analyze(
#                         cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
#                         actions=['emotion'],
#                         detector_backend=backend,
#                         enforce_detection=False,
#                         silent=True
#                     )
                    
#                     if results and 'emotion' in results[0]:
#                         emotions = results[0]['emotion']
#                         region = results[0]['region']
#                         x, y, w, h = region['x'], region['y'], region['w'], region['h']
                        
#                         # Get dominant emotion
#                         dominant = max(emotions.items(), key=lambda x: x[1])
                        
#                         # Draw face rectangle
#                         cv2.rectangle(frame, (x, y), (x+w, y+h), COLORS[dominant[0]], 2)
                        
#                         # Create dashboard
#                         dashboard = np.zeros((150, 400, 3), dtype=np.uint8)
                        
#                         # Add title
#                         cv2.putText(dashboard, "Emotion Analysis", (10, 30), 
#                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                        
#                         # Add dominant emotion
#                         cv2.putText(dashboard, f"Dominant: {dominant[0]} {dominant[1]:.1f}%", 
#                                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[dominant[0]], 1)
                        
#                         # Add all emotions
#                         y_offset = 90
#                         for emotion in EMOTIONS:
#                             percentage = emotions.get(emotion, 0)
#                             cv2.putText(dashboard, f"{emotion}: {percentage:.1f}%", 
#                                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[emotion], 1)
#                             y_offset += 20
                        
#                         # Overlay dashboard
#                         frame[10:160, 10:410] = dashboard
#                         break
                        
#                 except Exception as e:
#                     continue
                    
#         except Exception as e:
#             print(f"Analysis error: {str(e)}")
        
#         # Display instructions
#         cv2.putText(frame, "Make clear expressions", (10, frame.shape[0]-20), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
#         cv2.imshow('Emotion Detection', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
            
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()

###############################################################################################################################

# import cv2
# import numpy as np
# from deepface import DeepFace
# from collections import defaultdict

# # Enhanced emotion configuration with anger spectrum
# EMOTIONS = {
#     'angry': {
#         'subtypes': {
#             'annoyance': {'brow_wrinkles': 1, 'eye_tension': 1},
#             'frustration': {'brow_wrinkles': 2, 'jaw_clench': 1},
#             'resentment': {'lip_press': 2, 'brow_wrinkles': 1},
#             'indignation': {'eye_stare': 2, 'brow_wrinkles': 2},
#             'rage': {'mouth_open': 2, 'brow_wrinkles': 3, 'flushed': 1},
#             'contempt': {'lip_corner_tight': 1, 'head_tilt': 1}
#         },
#         'color': (0, 0, 255),
#         'description': "Eyebrows lowered, eyes glaring, mouth tense"
#     },
#     'disgust': {
#         'color': (0, 153, 0),
#         'description': "Nose wrinkled, upper lip raised"
#     },
#     'fear': {
#         'color': (255, 255, 0),
#         'description': "Eyes wide, eyebrows raised"
#     },
#     'happy': {
#         'color': (0, 255, 0),
#         'description': "Cheeks raised, crow's feet"
#     },
#     'sad': {
#         'color': (255, 0, 0),
#         'description': "Inner eyebrows raised, mouth corners down"
#     },
#     'surprise': {
#         'color': (0, 255, 255),
#         'description': "Eyebrows raised, jaw drop"
#     },
#     'neutral': {
#         'color': (255, 255, 255),
#         'description': "Relaxed facial muscles"
#     }
# }

# class AdvancedEmotionDetector:
#     def __init__(self):
#         self.emotion_counts = defaultdict(int)
#         self.anger_subtype_counts = defaultdict(int)
#         self.total_frames = 0
#         self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#         self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
#     def detect_anger_features(self, frame, face_region):
#         """Detect specific facial features of anger"""
#         x, y, w, h = face_region
#         roi = frame[y:y+h, x:x+w]
#         gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
#         anger_features = {
#             'brow_wrinkles': 0,  # Vertical wrinkles between brows
#             'eye_tension': 0,     # Eyelid tension
#             'jaw_clench': 0,      # Jaw muscle tension
#             'lip_press': 0,       # Lips pressed together
#             'mouth_open': 0,      # Mouth open aggressively
#             'eye_stare': 0,      # Hard gaze
#             'flushed': 0,        # Redness in face
#             'lip_corner_tight': 0 # Lip corners tightened
#         }
        
#         # Detect eyes for eye-related features
#         eyes = self.eye_cascade.detectMultiScale(gray, 1.3, 5)
        
#         if len(eyes) >= 2:
#             for (ex, ey, ew, eh) in eyes:
#                 eye_roi = roi[ey:ey+eh, ex:ex+ew]
                
#                 # Detect eye tension (glare)
#                 if cv2.Laplacian(eye_roi, cv2.CV_64F).var() > 150:
#                     anger_features['eye_tension'] += 1
#                     anger_features['eye_stare'] += 1
                
#                 # Detect eyelid narrowing
#                 if eh < h/5:  # Eyes appear more narrowed
#                     anger_features['eye_tension'] += 1
        
#         # Detect brow wrinkles (simplified)
#         brow_region = gray[0:int(h/3), :]
#         if cv2.Laplacian(brow_region, cv2.CV_64F).var() > 200:
#             anger_features['brow_wrinkles'] = min(3, anger_features['brow_wrinkles'] + 2)
        
#         # Detect mouth features
#         mouth_region = gray[int(h/2):h, :]
#         _, mouth_thresh = cv2.threshold(mouth_region, 150, 255, cv2.THRESH_BINARY)
        
#         # Check for pressed lips
#         if np.sum(mouth_thresh) < 5000:
#             anger_features['lip_press'] += 2
#         # Check for open mouth (rage)
#         elif np.sum(mouth_thresh) > 15000:
#             anger_features['mouth_open'] += 2
        
#         # Detect jaw clench (simplified)
#         jaw_region = gray[int(3*h/4):h, :]
#         if cv2.Laplacian(jaw_region, cv2.CV_64F).var() > 100:
#             anger_features['jaw_clench'] += 1
        
#         # Detect flushed face (redness)
#         b, g, r = cv2.split(roi)
#         if np.mean(r) > np.mean(g) + 20 and np.mean(r) > np.mean(b) + 20:
#             anger_features['flushed'] += 1
            
#         return anger_features
    
#     def determine_anger_subtype(self, anger_features):
#         """Classify the type of anger based on features"""
#         scores = {
#             'annoyance': anger_features['brow_wrinkles'] + anger_features['eye_tension'],
#             'frustration': anger_features['brow_wrinkles'] + anger_features['jaw_clench'],
#             'resentment': anger_features['lip_press'] + anger_features['brow_wrinkles'],
#             'indignation': anger_features['eye_stare'] + anger_features['brow_wrinkles'],
#             'rage': anger_features['mouth_open'] + anger_features['brow_wrinkles'] + anger_features['flushed'],
#             'contempt': anger_features['lip_corner_tight']  # + other contempt-specific features
#         }
#         return max(scores.items(), key=lambda x: x[1])[0]
    
#     def process_frame(self, frame):
#         self.total_frames += 1
        
#         try:
#             # Get DeepFace analysis
#             results = DeepFace.analyze(
#                 cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
#                 actions=['emotion'],
#                 detector_backend='mtcnn',
#                 enforce_detection=False,
#                 silent=True
#             )
            
#             if not results or 'emotion' not in results[0]:
#                 return frame
                
#             result = results[0]
#             emotions = result['emotion']
#             region = result['region']
#             x, y, w, h = region['x'], region['y'], region['w'], region['h']
            
#             # Special handling for anger
#             if emotions['angry'] > 25:  # Minimum anger threshold
#                 anger_features = self.detect_anger_features(frame, (x, y, w, h))
#                 anger_subtype = self.determine_anger_subtype(anger_features)
#                 self.anger_subtype_counts[anger_subtype] += 1
#                 emotions['angry'] = min(100, emotions['angry'] + 15)  # Boost confidence
            
#             # Update counts
#             dominant = max(emotions.items(), key=lambda x: x[1])
#             if dominant[1] > 25:  # Minimum confidence
#                 self.emotion_counts[dominant[0]] += 1
            
#             # Draw results
#             self.draw_analysis(frame, emotions, (x, y, w, h))
            
#         except Exception as e:
#             print(f"Processing error: {str(e)}")
            
#         return frame
    
#     def draw_analysis(self, frame, emotions, face_region):
#         x, y, w, h = face_region
#         dominant = max(emotions.items(), key=lambda x: x[1])
#         color = EMOTIONS.get(dominant[0], {}).get('color', (255, 255, 255))
        
#         # Draw face rectangle
#         cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
#         # Create dashboard
#         dashboard_height = 200
#         dashboard_width = 400
#         dashboard = np.zeros((dashboard_height, dashboard_width, 3), dtype=np.uint8)
        
#         # Add title
#         cv2.putText(dashboard, "Emotion Analysis", (10, 30), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
#         # Add dominant emotion
#         cv2.putText(dashboard, f"Dominant: {dominant[0]} {dominant[1]:.1f}%", 
#                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        
#         # Add anger subtype if applicable
#         if dominant[0] == 'angry' and self.anger_subtype_counts:
#             current_subtype = max(self.anger_subtype_counts.items(), key=lambda x: x[1])[0]
#             cv2.putText(dashboard, f"Anger Type: {current_subtype}", 
#                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
#         # Add all emotions
#         y_offset = 120
#         for emotion, config in EMOTIONS.items():
#             percent = emotions.get(emotion, 0)
#             bar_width = int(percent * 3)
            
#             cv2.putText(dashboard, f"{emotion[:8]:<8}: {percent:5.1f}%", 
#                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, config['color'], 1)
#             cv2.rectangle(dashboard, (150, y_offset-5), (150+bar_width, y_offset+5), config['color'], -1)
#             y_offset += 25
        
#         # Overlay dashboard
#         frame[10:10+dashboard_height, 10:10+dashboard_width] = dashboard

# def main():
#     detector = AdvancedEmotionDetector()
#     cap = cv2.VideoCapture(0)
    
#     if not cap.isOpened():
#         print("Error: Could not open webcam.")
#         return
    
#     # Set resolution
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
#     print("Press 'Q' to quit")
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
            
#         frame = cv2.flip(frame, 1)
#         frame = detector.process_frame(frame)
        
#         cv2.imshow('Advanced Emotion Detection', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
            
#     cap.release()
#     cv2.destroyAllWindows()
    
#     # Print final report
#     print("\n=== Final Emotion Report ===")
#     print(f"Total frames processed: {detector.total_frames}")
    
#     print("\nMain Emotions:")
#     for emotion in EMOTIONS:
#         count = detector.emotion_counts.get(emotion, 0)
#         percentage = (count / detector.total_frames) * 100 if detector.total_frames > 0 else 0
#         print(f"{emotion.capitalize():<10}: {count:>4} frames ({percentage:.1f}%)")
    
#     if detector.anger_subtype_counts:
#         print("\nAnger Subtypes:")
#         for subtype, count in detector.anger_subtype_counts.items():
#             percentage = (count / detector.total_frames) * 100 if detector.total_frames > 0 else 0
#             print(f"{subtype.capitalize():<12}: {count:>4} frames ({percentage:.1f}%)")

# if __name__ == "__main__":
#     main()

################################################################################################################################\
    
# import cv2
# import numpy as np
# from deepface import DeepFace
# from collections import defaultdict

# # Comprehensive emotion configuration with disgust spectrum
# EMOTIONS = {
#     'angry': {
#         'subtypes': {
#             'annoyance': {'brow_wrinkles': 1, 'eye_tension': 1},
#             'frustration': {'brow_wrinkles': 2, 'jaw_clench': 1},
#             'resentment': {'lip_press': 2, 'brow_wrinkles': 1},
#             'indignation': {'eye_stare': 2, 'brow_wrinkles': 2},
#             'rage': {'mouth_open': 2, 'brow_wrinkles': 3, 'flushed': 1},
#             'contempt': {'lip_corner_tight': 1, 'head_tilt': 1}
#         },
#         'color': (0, 0, 255),
#         'description': "Eyebrows lowered, eyes glaring, mouth tense"
#     },
#     'disgust': {
#         'subtypes': {
#             'aversion': {'nose_wrinkle': 1, 'upper_lip_raise': 1},
#             'revulsion': {'nose_wrinkle': 2, 'upper_lip_raise': 2, 'head_turn': 1},
#             'repugnance': {'nose_wrinkle': 3, 'upper_lip_raise': 2, 'eye_squint': 1},
#             'loathing': {'nose_wrinkle': 3, 'upper_lip_raise': 3, 'brow_lower': 2},
#             'contempt': {'lip_corner_tight': 2, 'asymmetry': 1}
#         },
#         'color': (0, 153, 0),
#         'description': "Nose wrinkled, upper lip raised"
#     },
#     'fear': {
#         'color': (255, 255, 0),
#         'description': "Eyes wide, eyebrows raised"
#     },
#     'happy': {
#         'color': (0, 255, 0),
#         'description': "Cheeks raised, crow's feet"
#     },
#     'sad': {
#         'color': (255, 0, 0),
#         'description': "Inner eyebrows raised, mouth corners down"
#     },
#     'surprise': {
#         'color': (0, 255, 255),
#         'description': "Eyebrows raised, jaw drop"
#     },
#     'neutral': {
#         'color': (255, 255, 255),
#         'description': "Relaxed facial muscles"
#     }
# }

# class CompleteEmotionDetector:
#     def __init__(self):
#         self.emotion_counts = defaultdict(int)
#         self.subtype_counts = defaultdict(lambda: defaultdict(int))
#         self.total_frames = 0
#         self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
#     def detect_disgust_features(self, frame, face_region):
#         """Detect specific facial features of disgust"""
#         x, y, w, h = face_region
#         roi = frame[y:y+h, x:x+w]
#         gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
#         disgust_features = {
#             'nose_wrinkle': 0,      # Nose scrunching (AU9)
#             'upper_lip_raise': 0,    # Upper lip raised (AU10)
#             'brow_lower': 0,         # Eyebrows lowered (AU4)
#             'eye_squint': 0,         # Eyes squinting
#             'head_turn': 0,          # Head turning away
#             'asymmetry': 0,          # For contempt detection
#             'lip_corner_tight': 0    # Lip corner tightening
#         }
        
#         # Detect nose wrinkles (primary disgust feature)
#         nose_region = gray[int(h/4):int(h/2), int(w/4):int(3*w/4)]
#         if cv2.Laplacian(nose_region, cv2.CV_64F).var() > 200:
#             disgust_features['nose_wrinkle'] = 2  # Strong nose wrinkle
            
#         # Detect upper lip raise
#         upper_lip_region = gray[int(h/2):int(2*h/3), :]
#         _, upper_lip_thresh = cv2.threshold(upper_lip_region, 150, 255, cv2.THRESH_BINARY_INV)
#         if np.sum(upper_lip_thresh) > 5000:
#             disgust_features['upper_lip_raise'] = 2
            
#         # Detect brow lowering
#         brow_region = gray[0:int(h/3), :]
#         if cv2.Laplacian(brow_region, cv2.CV_64F).var() > 150:
#             disgust_features['brow_lower'] = 1
            
#         # Detect eye squint
#         eye_region = gray[int(h/5):int(h/2), :]
#         if cv2.Laplacian(eye_region, cv2.CV_64F).var() > 180:
#             disgust_features['eye_squint'] = 1
            
#         # Detect contempt features (asymmetry)
#         left_side = gray[:, :int(w/2)]
#         right_side = gray[:, int(w/2):]
#         if abs(cv2.Laplacian(left_side, cv2.CV_64F).var() - cv2.Laplacian(right_side, cv2.CV_64F).var()) > 50:
#             disgust_features['asymmetry'] = 1
#             disgust_features['lip_corner_tight'] = 1
            
#         return disgust_features
    
#     def determine_disgust_subtype(self, disgust_features):
#         """Classify the type of disgust based on features"""
#         scores = {
#             'aversion': disgust_features['nose_wrinkle'] + disgust_features['upper_lip_raise'],
#             'revulsion': disgust_features['nose_wrinkle'] + disgust_features['upper_lip_raise'] + disgust_features['head_turn'],
#             'repugnance': disgust_features['nose_wrinkle'] + disgust_features['upper_lip_raise'] + disgust_features['eye_squint'],
#             'loathing': disgust_features['nose_wrinkle'] + disgust_features['upper_lip_raise'] + disgust_features['brow_lower'],
#             'contempt': disgust_features['lip_corner_tight'] + disgust_features['asymmetry']
#         }
#         return max(scores.items(), key=lambda x: x[1])[0]
    
#     def process_frame(self, frame):
#         self.total_frames += 1
        
#         try:
#             # Get DeepFace analysis
#             results = DeepFace.analyze(
#                 cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
#                 actions=['emotion'],
#                 detector_backend='mtcnn',
#                 enforce_detection=False,
#                 silent=True
#             )
            
#             if not results or 'emotion' not in results[0]:
#                 return frame
                
#             result = results[0]
#             emotions = result['emotion']
#             region = result['region']
#             x, y, w, h = region['x'], region['y'], region['w'], region['h']
            
#             # Special handling for anger and disgust
#             if emotions['angry'] > 25:
#                 anger_features = self.detect_anger_features(frame, (x, y, w, h))
#                 anger_subtype = self.determine_anger_subtype(anger_features)
#                 self.subtype_counts['angry'][anger_subtype] += 1
#                 emotions['angry'] = min(100, emotions['angry'] + 15)
                
#             if emotions['disgust'] > 25:
#                 disgust_features = self.detect_disgust_features(frame, (x, y, w, h))
#                 disgust_subtype = self.determine_disgust_subtype(disgust_features)
#                 self.subtype_counts['disgust'][disgust_subtype] += 1
#                 emotions['disgust'] = min(100, emotions['disgust'] + 15)
            
#             # Update counts
#             dominant = max(emotions.items(), key=lambda x: x[1])
#             if dominant[1] > 25:
#                 self.emotion_counts[dominant[0]] += 1
            
#             # Draw results
#             self.draw_analysis(frame, emotions, (x, y, w, h))
            
#         except Exception as e:
#             print(f"Processing error: {str(e)}")
            
#         return frame
    
#     def draw_analysis(self, frame, emotions, face_region):
#         x, y, w, h = face_region
#         dominant = max(emotions.items(), key=lambda x: x[1])
#         color = EMOTIONS.get(dominant[0], {}).get('color', (255, 255, 255))
        
#         # Draw face rectangle
#         cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
#         # Create dashboard
#         dashboard_height = 220
#         dashboard_width = 450
#         dashboard = np.zeros((dashboard_height, dashboard_width, 3), dtype=np.uint8)
        
#         # Add title
#         cv2.putText(dashboard, "Advanced Emotion Analysis", (10, 30), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
#         # Add dominant emotion
#         cv2.putText(dashboard, f"Dominant: {dominant[0]} {dominant[1]:.1f}%", 
#                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        
#         # Add subtype if applicable
#         if dominant[0] in ['angry', 'disgust'] and self.subtype_counts.get(dominant[0]):
#             current_subtype = max(self.subtype_counts[dominant[0]].items(), key=lambda x: x[1])[0]
#             cv2.putText(dashboard, f"Type: {current_subtype}", 
#                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
#         # Add all emotions
#         y_offset = 120
#         for emotion, config in EMOTIONS.items():
#             percent = emotions.get(emotion, 0)
#             bar_width = int(percent * 3)
            
#             cv2.putText(dashboard, f"{emotion[:8]:<8}: {percent:5.1f}%", 
#                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, config['color'], 1)
#             cv2.rectangle(dashboard, (150, y_offset-5), (150+bar_width, y_offset+5), config['color'], -1)
            
#             # Add feature indicators for disgust and anger
#             if emotion == dominant[0] and emotion in ['disgust', 'angry']:
#                 features = self.get_active_features(emotion, face_region)
#                 cv2.putText(dashboard, f"Features: {', '.join(features)}", 
#                            (250, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
#             y_offset += 25
        
#         # Overlay dashboard
#         frame[10:10+dashboard_height, 10:10+dashboard_width] = dashboard
    
#     def get_active_features(self, emotion, face_region):
#         """Get active facial features for the current emotion"""
#         if emotion == 'disgust':
#             features = self.detect_disgust_features(frame, face_region)
#             return [k for k, v in features.items() if v > 0]
#         elif emotion == 'angry':
#             features = self.detect_anger_features(frame, face_region)
#             return [k for k, v in features.items() if v > 0]
#         return []

# def main():
#     detector = CompleteEmotionDetector()
#     cap = cv2.VideoCapture(0)
    
#     if not cap.isOpened():
#         print("Error: Could not open webcam.")
#         return
    
#     # Set resolution
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
#     print("Press 'Q' to quit")
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
            
#         frame = cv2.flip(frame, 1)
#         frame = detector.process_frame(frame)
        
#         cv2.imshow('Complete Emotion Detection', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
            
#     cap.release()
#     cv2.destroyAllWindows()
    
#     # Print final report
#     print("\n=== Final Emotion Report ===")
#     print(f"Total frames processed: {detector.total_frames}")
    
#     print("\nMain Emotions:")
#     for emotion in EMOTIONS:
#         count = detector.emotion_counts.get(emotion, 0)
#         percentage = (count / detector.total_frames) * 100 if detector.total_frames > 0 else 0
#         print(f"{emotion.capitalize():<10}: {count:>4} frames ({percentage:.1f}%)")
    
#     for emotion in ['angry', 'disgust']:
#         if detector.subtype_counts.get(emotion):
#             print(f"\n{emotion.capitalize()} Subtypes:")
#             for subtype, count in detector.subtype_counts[emotion].items():
#                 percentage = (count / detector.total_frames) * 100 if detector.total_frames > 0 else 0
#                 print(f"{subtype.capitalize():<12}: {count:>4} frames ({percentage:.1f}%)")

# if __name__ == "__main__":
#     main()
######################################################################################################################################
### main code 
import cv2
import numpy as np
from deepface import DeepFace
from collections import defaultdict

# Comprehensive emotion configuration with subtypes for multiple emotions
EMOTIONS = {
    'angry': {
        'subtypes': {
            'annoyance': {'brow_furrow': 1, 'eye_squint': 1},
            'frustration': {'brow_furrow': 2, 'jaw_clench': 1, 'lip_press': 1},
            'rage': {'mouth_open': 2, 'brow_furrow': 3, 'eye_squint': 2},
        },
        'color': (0, 0, 255),
        'description': "Eyebrows lowered, eyes glaring, mouth tense"
    },
    'happy': {
        'subtypes': {
            'amusement': {'mouth_smile': 1},
            'joy': {'mouth_smile': 2, 'eye_crinkle': 1},
            'elation': {'mouth_smile': 3, 'eye_crinkle': 2},
        },
        'color': (0, 255, 0),
        'description': "Cheeks raised, mouth corners up"
    },
    'sad': {
        'subtypes': {
            'disappointment': {'mouth_frown': 1},
            'sorrow': {'mouth_frown': 2, 'brow_raise': 1},
            'grief': {'mouth_frown': 3, 'brow_raise': 2},
        },
        'color': (255, 0, 0),
        'description': "Inner eyebrows raised, mouth corners down"
    },
    'fear': {
        'subtypes': {
            'anxiety': {'eye_widen': 1},
            'alarm': {'eye_widen': 2, 'mouth_open': 1},
            'terror': {'eye_widen': 3, 'mouth_open': 2},
        },
        'color': (255, 255, 0),
        'description': "Eyes wide, eyebrows raised, mouth open"
    },
    'disgust': {'color': (0, 153, 0), 'description': "Nose wrinkled, upper lip raised"},
    'surprise': {'color': (0, 255, 255), 'description': "Eyebrows raised, jaw drop"},
    'neutral': {'color': (255, 255, 255), 'description': "Relaxed facial muscles"}
}

class CompleteEmotionDetector:
    def __init__(self):
        self.emotion_counts = defaultdict(int)
        self.subtype_counts = defaultdict(lambda: defaultdict(int))
        self.total_frames = 0
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # --- Feature Detectors for Each Emotion ---

    def detect_anger_features(self, roi_gray, w, h):
        features = {'brow_furrow': 0, 'eye_squint': 0, 'jaw_clench': 0, 'lip_press': 0, 'mouth_open': 0}
        
        # Brow Furrow: Look for vertical edges between eyebrows
        brow_region = roi_gray[h//5:h//2, w//3:2*w//3]
        sobel_x = cv2.Sobel(brow_region, cv2.CV_64F, 1, 0, ksize=3)
        if np.mean(np.abs(sobel_x)) > 20: features['brow_furrow'] = 2

        # Eye Squint: Check for small eye aspect ratio
        eyes = self.eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
        if len(eyes) > 0:
            avg_eye_height = np.mean([eh for (ex, ey, ew, eh) in eyes])
            if (avg_eye_height / h) < 0.1: features['eye_squint'] = 2

        # Mouth State: Use contour area
        mouth_roi = roi_gray[2*h//3:h, w//4:3*w//4]
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
        
        # Mouth Smile: Check for wide aspect ratio of the mouth contour
        mouth_roi = roi_gray[2*h//3:h, w//5:4*w//5]
        thresh = cv2.adaptiveThreshold(mouth_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x_c, y_c, w_c, h_c = cv2.boundingRect(largest_contour)
            if w_c > 0 and h_c > 0 and (w_c / h_c) > 2.0:
                features['mouth_smile'] = 2
                
        # Eye Crinkle (Crow's Feet): Use Laplacian variance to detect fine wrinkles
        eyes = self.eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
        if len(eyes) > 0:
            eye_crinkle_variance = 0
            for (ex, ey, ew, eh) in eyes:
                # Check region to the side of the eye
                corner_roi = roi_gray[ey:ey+eh, max(0, ex-ew//2):ex]
                if corner_roi.size > 0:
                    eye_crinkle_variance += cv2.Laplacian(corner_roi, cv2.CV_64F).var()
            if eye_crinkle_variance / len(eyes) > 100: # High texture indicates wrinkles
                features['eye_crinkle'] = 2
        return features
        
    def detect_sad_features(self, roi_gray, w, h):
        features = {'mouth_frown': 0, 'brow_raise': 0}
        
        # Mouth Frown: Check for downward curve
        mouth_roi = roi_gray[2*h//3:h, w//4:3*w//4]
        thresh = cv2.adaptiveThreshold(mouth_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            x_c, y_c, w_c, h_c = cv2.boundingRect(cnt)
            # For a frown, the lower part of the bounding box will have more contour points
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cy = int(M['m01'] / M['m00'])
                if (cy / h_c) > 0.6: # Centroid is in the lower 40% of the box
                    features['mouth_frown'] = 2

        # Inner Brow Raise: Look for increased vertical space and texture above nose
        brow_center_roi = roi_gray[h//5:h//2, w//3:2*w//3]
        if cv2.Laplacian(brow_center_roi, cv2.CV_64F).var() > 60:
            features['brow_raise'] = 1
        return features

    def detect_fear_features(self, roi_gray, w, h):
        features = {'eye_widen': 0, 'mouth_open': 0}

        # Wide Eyes: Check for a more circular shape (aspect ratio close to 1)
        eyes = self.eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
        if len(eyes) > 0:
            aspect_ratios = []
            for (ex, ey, ew, eh) in eyes:
                if ew > 0:
                    aspect_ratios.append(eh / ew)
            avg_ratio = np.mean(aspect_ratios)
            if avg_ratio > 0.8: # Eyes are very round
                features['eye_widen'] = 2

        # Mouth Open (Jaw Drop): Check for tall mouth shape
        mouth_roi = roi_gray[2*h//3:h, w//4:3*w//4]
        thresh = cv2.adaptiveThreshold(mouth_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            x_c, y_c, w_c, h_c = cv2.boundingRect(cnt)
            if h_c > 0 and (w_c / h_c) < 1.5: # Mouth is taller/more circular than wide
                features['mouth_open'] = 2
        return features

    def determine_subtype(self, emotion, features):
        """Generic subtype determination based on feature scores."""
        scores = defaultdict(int)
        if emotion not in EMOTIONS or 'subtypes' not in EMOTIONS[emotion]:
            return ""

        for subtype, feature_map in EMOTIONS[emotion]['subtypes'].items():
            score = 0
            for feature, weight in feature_map.items():
                if features.get(feature, 0) > 0:
                    score += features[feature] * weight
            scores[subtype] = score
        
        if not any(scores.values()):
            return list(EMOTIONS[emotion]['subtypes'].keys())[0] # Default to first subtype
        return max(scores.items(), key=lambda x: x[1])[0]

    def process_frame(self, frame):
        self.total_frames += 1
        
        try:
            results = DeepFace.analyze(
                frame, actions=['emotion'],
                detector_backend='opencv', enforce_detection=False, silent=True
            )
            
            result = results[0]
            emotions = result['emotion']
            region = result['region']
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            
            roi_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            dominant_emotion = result['dominant_emotion']
            
            feature_detectors = {
                'angry': self.detect_anger_features,
                'happy': self.detect_happy_features,
                'sad': self.detect_sad_features,
                'fear': self.detect_fear_features
            }

            if dominant_emotion in feature_detectors:
                features = feature_detectors[dominant_emotion](roi_gray, w, h)
                subtype = self.determine_subtype(dominant_emotion, features)
                self.subtype_counts[dominant_emotion][subtype] += 1

            self.emotion_counts[dominant_emotion] += 1
            self.draw_analysis(frame, emotions, (x, y, w, h))
            
        except Exception as e:
            pass # Suppress errors in live feed
            
        return frame

    def draw_analysis(self, frame, emotions, face_region):
        x, y, w, h = face_region
        dominant = max(emotions.items(), key=lambda x: x[1])
        dominant_emotion = dominant[0]
        color = EMOTIONS.get(dominant_emotion, {}).get('color', (255, 255, 255))
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        dashboard_height, dashboard_width = 220, 300
        dashboard = np.zeros((dashboard_height, dashboard_width, 3), dtype=np.uint8)
        
        cv2.putText(dashboard, "Emotion Analysis", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(dashboard, f"Dominant: {dominant_emotion.capitalize()}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        # Display subtype if available
        if dominant_emotion in self.subtype_counts and self.subtype_counts[dominant_emotion]:
             current_subtype = max(self.subtype_counts[dominant_emotion].items(), key=lambda x: x[1])[0]
             cv2.putText(dashboard, f"Type: {current_subtype.capitalize()}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Display emotion bars
        y_offset = 110
        for emotion, percent in sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:5]:
            bar_width = int((percent / 100) * (dashboard_width - 110))
            emo_color = EMOTIONS.get(emotion, {}).get('color', (255, 255, 255))
            cv2.putText(dashboard, f"{emotion.capitalize()}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, emo_color, 1)
            cv2.rectangle(dashboard, (100, y_offset-10), (100+bar_width, y_offset), emo_color, -1)
            y_offset += 22
            
        frame[10:10+dashboard_height, 10:10+dashboard_width] = dashboard

def main():
    detector = CompleteEmotionDetector()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("Press 'Q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
            
        frame = cv2.flip(frame, 1)
        frame = detector.process_frame(frame)
        
        cv2.imshow('Complete Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final report
    print("\n=== Final Emotion Report ===")
    print(f"Total frames processed: {detector.total_frames}")
    
    print("\n--- Main Emotion Distribution ---")
    for emotion, count in detector.emotion_counts.items():
        percentage = (count / detector.total_frames) * 100 if detector.total_frames > 0 else 0
        print(f"{emotion.capitalize():<10}: {count:>4} frames ({percentage:.1f}%)")
    
    print("\n--- Subtype Distribution ---")
    for emotion, subtypes in detector.subtype_counts.items():
        if subtypes:
            print(f"\n{emotion.capitalize()} Subtypes:")
            total_subtype_frames = sum(subtypes.values())
            for subtype, count in subtypes.items():
                percentage = (count / total_subtype_frames) * 100 if total_subtype_frames > 0 else 0
                print(f"  - {subtype.capitalize():<12}: {count:>4} frames ({percentage:.1f}%)")

if __name__ == "__main__":
    main()
    
#################################################################################################################################