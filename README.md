# Mental-Burnout-Detection-System

**Tagline:**
A multi-modal system using facial, vocal, and questionnaire-based analysis to detect early signs of mental burnout.

## 1. Introduction

Mental burnout is a growing concern among students, professionals, and healthcare workers. It leads to decreased productivity, poor mental health, and long-term health issues.
Our project addresses this by creating a multi-modal burnout detection system that integrates:

- **Questionnaires** for self-reported symptoms.

- **Facial Analysis** (eye and mouth fatigue indicators + emotion recognition).

- **Voice Analysis** (sentiment, transcription, and pitch variation).


This holistic approach enables more accurate, real-time detection of burnout signs compared to single-modality systems.

## 2. Features

- Interactive web-based questionnaire for self-assessment.

- Real-time facial analysis using EAR (Eye Aspect Ratio), MAR (Mouth Aspect Ratio), and emotion detection.

- Voice-based analysis for sentiment and pitch fluctuations.

- Speech-to-text transcription for further semantic understanding.

- Data storage for further training and model improvement.

- Modular design for easy scalability and future integration with healthcare platforms.

 ## 3. Technical Details and Methodology

### Questionnaire

- Collects user-reported stress and fatigue levels.

- Based on validated clinical burnout scales ([Burnout scale](https://link.springer.com/article/10.1007/s11606-014-3112-6)).


### Facial Analysis

- **EAR (Eye Aspect Ratio):** Detects eye closure, drowsiness, and fatigue .([source]( https://www.mdpi.com/1424-8220/24/17/5683)).

- **MAR (Mouth Aspect Ratio):** Captures yawning/fatigue signals.

- **Emotion Detection:** Uses DeepFace for classifying facial emotions (happy, sad, stressed, angry, etc.) ([source]( https://arxiv.org/abs/2504.03010)).


 ### Voice Analysis

- **Speech-to-Text:** Converts audio input into text for further sentiment analysis.

- **Sentiment Analysis:** Implements VADER sentiment scoring on transcripts.


 ## 4. Research References

1.** Burnout Questionnaire Scale:** [Spring - Internal Medicine Burnout study](https://link.springer.com/article/10.1007/s11606-014-3112-6)


2. **EAR & MAR Metrics:**[MDPI- Eye/ Mouth Aspect Ratio in Fatigue Detection](https://www.mdpi.com/1424-8220/24/17/56830)


3. **Emotion Detection & Intelligence:** [arXiv - Emotional Intelligence via Deep Learning](https://arxiv.org/abs/2504.03010)


4. **Voice Stress & Sentiment:** [AJCST - Speech Processing for Stress Detection](https://www.ajcst.co/index.php/ajcst/article/view/2037/6786)

## 5. Getting Started

### Prerequisites

Hardware: Webcam & Microphone.

Software:Python 3.8+ , pip

 ### Installation
''' bash
# Clone repository
git clone
https://github.com/your-username/mental-burnout-detector.git
cd mental-burnout-detector

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py

## File Structure

.
├── app.py                     
├── deepface.py                
├── ear_mar_calculation.py     
├── voice.py                   
├── requirements.txt           
└── README.md                  

## Contributions

- Muskan Bisen (Lead Developer & Full-Stack): Website UI/UX, EAR/MAR logic integration.

- Team Member 2 (Voice Analysis Lead): Implemented voice sentiment & pitch analysis, audio pipeline.

- Team Member 3 (AI Integration & Management): Integrated emotion detection mod.el, coordinated project development



