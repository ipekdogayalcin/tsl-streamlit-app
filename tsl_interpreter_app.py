"""
SignSpeak: Real-Time Turkish Sign Language Recognition Web Application

This Streamlit application implements real-time Turkish Sign Language recognition using
a pre-trained LSTM model and MediaPipe for pose estimation. The app supports both
webcam input and video file uploads for sign language recognition.

Key Features:
•⁠  ⁠Real-time webcam processing
•⁠  ⁠Video file upload support
•⁠  ⁠Natural language processing with transformers
•⁠  ⁠Dynamic sentence generation
•⁠  ⁠Smooth prediction using rolling average
•⁠  ⁠User-friendly interface with Streamlit

Author: Merve Kahraman & İpek Doğa Yalçın & Seher Zeynep Sonkaya
Date: December 2024
"""

import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import mediapipe as mp
from collections import deque
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
import torch
from typing import List, Dict
import time

# Initialize the Turkish BERT model and tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
    model = AutoModelForMaskedLM.from_pretrained("dbmdz/bert-base-turkish-cased")
    model = model.to('cpu')  # Explicitly move to CPU
    fill_mask_pipeline = pipeline("fill-mask", model=model, tokenizer=tokenizer, device=-1)  # Use CPU
except Exception as e:
    st.error(f"Error loading NLP model: {str(e)}")
    st.info("Please run: pip install transformers torch")
    fill_mask_pipeline = None

# Define constants
max_seq_length = 30
threshold = 0.5

# Full CLASS_MAP as provided by user
CLASS_MAP = {
    0: "abla (sister)", 1: "acele (hurry)", 2: "acikmak (hungry)",
    3: "afiyet_olsun (enjoy_your_meal)", 4: "agabey (brother)",
    5: "agac (tree)", 6: "agir (heavy)", 7: "aglamak (cry)",
    8: "aile (family)", 9: "akilli (wise)", 10: "akilsiz (unwise)",
    11: "akraba (kin)", 12: "alisveris (shopping)", 13: "anahtar (key)",
    14: "anne (mother)", 15: "arkadas (friend)", 16: "ataturk (ataturk)",
    17: "ayakkabi (shoe)", 18: "ayna (mirror)", 19: "ayni (same)",
    20: "baba (father)", 21: "bahce (garden)", 22: "bakmak (look)",
    23: "bal (honey)", 24: "bardak (glass)", 25: "bayrak (flag)",
    26: "bayram (feast)", 27: "bebek (baby)", 28: "bekar (single)",
    29: "beklemek (wait)", 30: "ben (I)", 31: "benzin (petrol)",
    32: "beraber (together)", 33: "bilgi_vermek (inform)", 34: "biz (we)",
    35: "calismak (work)", 36: "carsamba (wednesday)", 37: "catal (fork)",
    38: "cay (tea)", 39: "caydanlik (teapot)", 40: "cekic (hammer)",
    41: "cirkin (ugly)", 42: "cocuk (child)", 43: "corba (soup)",
    44: "cuma (friday)", 45: "cumartesi (saturday)", 46: "cuzdan (wallet)",
    47: "dakika (minute)", 48: "dede (grandfather)", 49: "degistirmek (change)",
    50: "devirmek (topple)", 51: "devlet (government)", 52: "doktor (doctor)",
    53: "dolu (full)", 54: "dugun (wedding)", 55: "dun (yesterday)",
    56: "dusman (enemy)", 57: "duvar (wall)", 58: "eczane (pharmacy)",
    59: "eldiven (glove)", 60: "emek (labor)", 61: "emekli (retired)",
    62: "erkek (male)", 63: "et (meal)", 64: "ev (house)", 65: "evet (yes)",
    66: "evli (married)", 67: "ezberlemek (memorize)", 68: "fil (elephant)",
    69: "fotograf (photograph)", 70: "futbol (football)", 71: "gecmis (past)",
    72: "gecmis_olsun (get_well)", 73: "getirmek (bring)", 74: "gol (lake)",
    75: "gomlek (shirt)", 76: "gormek (see)", 77: "gostermek (show)",
    78: "gulmek (laugh)", 79: "hafif (lightweight)", 80: "hakli (right)",
    81: "hali (carpet)", 82: "hasta (ill)", 83: "hastane (hospital)",
    84: "hata (fault)", 85: "havlu (towel)", 86: "hayir (no)",
    87: "hayirli_olsun (congratulations)", 88: "hayvan (animal)",
    89: "hediye (gift)", 90: "helal (halal)", 91: "hep (always)",
    92: "hic (never)", 93: "hoscakal (goodbye)", 94: "icmek (drink)",
    95: "igne (needle)", 96: "ilac (medicine)", 97: "ilgilenmemek (not_interested)",
    98: "isik (light)", 99: "itmek (push)", 100: "iyi (good)",
    101: "kacmak (escape)", 102: "kahvalti (breakfast)", 103: "kalem (pencil)",
    104: "kalorifer (radiator)", 105: "kapi (door)", 106: "kardes (sibling)",
    107: "kavsak (crossroads)", 108: "kaza (accident)", 109: "kemer (belt)",
    110: "keske (if_only)", 111: "kim (who)", 112: "kimlik (identity)",
    113: "kira (rent)", 114: "kitap (book)", 115: "kiyma (mince)",
    116: "kiz (female)", 117: "koku (smell)", 118: "kolonya (cologne)",
    119: "komur (coal)", 120: "kopek (dog)", 121: "kopru (bridge)",
    122: "kotu (bad)", 123: "kucak (lap)", 124: "leke (stain)",
    125: "maas (salary)", 126: "makas (scissors)", 127: "masa (tongs)",
    128: "masallah (god_preserve)", 129: "melek (angel)", 130: "memnun_olmak (be_pleased)",
    131: "mendil (napkin)", 132: "merdiven (stairs)", 133: "misafir (guest)",
    134: "mudur (manager)", 135: "musluk (tap)", 136: "nasil (how)",
    137: "neden (why)", 138: "nerede (where)", 139: "nine (grandmother)",
    140: "ocak (oven)", 141: "oda (room)", 142: "odun (wood)",
    143: "ogretmen (teacher)", 144: "okul (school)", 145: "olimpiyat (olympiad)",
    146: "olmaz (nope)", 147: "olur (allright)", 148: "onlar (they)",
    149: "orman (forest)", 150: "oruc (fasting)", 151: "ozur_dilemek (apologize)",
    152: "pamuk (cotton)", 153: "pantolon (trousers)", 154: "para (money)",
    155: "pastirma (pastrami)", 156: "patates (potato)", 157: "pazar (sunday)",
    158: "pazartesi (monday)", 159: "pencere (window)", 160: "persembe (thursday)",
    161: "piknik (picnic)", 162: "polis (police)", 163: "psikoloji (psychology)",
    164: "rica_etmek (request)", 165: "saat (hour)", 166: "sabun (soap)",
    167: "salca (sauce)", 168: "sali (tuesday)", 169: "sampiyon (champion)",
    170: "sapka (hat)", 171: "savas (war)", 172: "seker (sugar)",
    173: "selam (hi)", 174: "semsiye (umbrella)", 175: "sen (you)",
    176: "senet (bill)", 177: "serbest (free)", 178: "ses (voice)",
    179: "sevmek (love)", 180: "seytan (evil)", 181: "sinir (border)",
    182: "siz (you)", 183: "soylemek (say)", 184: "soz (promise)",
    185: "sut (milk)", 186: "tamam (okay)", 187: "tarak (comb)",
    188: "tarih (date)", 189: "tatil (holiday)", 190: "tatli (sweet)",
    191: "tavan (ceiling)", 192: "tehlike (danger)", 193: "telefon (telephone)",
    194: "terazi (scales)", 195: "terzi (tailor)", 196: "tesekkur (thanks)",
    197: "tornavida (screwdriver)", 198: "turkiye (turkey)", 199: "turuncu (orange)",
    200: "tuvalet (toilet)", 201: "un (flour)", 202: "uzak (far)",
    203: "uzgun (sad)", 204: "var (existing)", 205: "vergi (tax)",
    206: "yakin (near)", 207: "yalniz (alone)", 208: "yanlis (wrong)",
    209: "yapmak (do)", 210: "yarabandi (band-aid)", 211: "yardim (help)",
    212: "yarin (tomorrow)", 213: "yasak (forbidden)", 214: "yastik (pillow)",
    215: "yatak (bed)", 216: "yavas (slow)", 217: "yemek (eat)",
    218: "yemek_pisirmek (cook)", 219: "yildiz (star)", 220: "yok (absent)",
    221: "yol (road)", 222: "yorgun (tired)", 223: "yumurta (egg)",
    224: "zaman (time)", 225: "zor (difficult)"
}

# Helper: Turkish word type detection
PRONOUNS = {"ben", "biz", "sen", "siz", "onlar"}

# Helper: get word and type

def get_word_and_type(idx):
    label = CLASS_MAP.get(idx, "")
    if not label:
        return "", "OTHER"
    word = label.split()[0].replace("_", " ")
    # Heuristic: verbs end with 'mek' or 'mak'
    if word.endswith("mek") or word.endswith("mak"):
        return word, "VERB"
    if word in PRONOUNS:
        return word, "SUBJECT"
    return word, "OBJECT"

# NLP Processor for all words
class TurkishNLPProcessor:
    def __init__(self):
        pass
    def generate_sentence(self, gesture_sequence):
        if not gesture_sequence:
            return ""
        words_and_types = [get_word_and_type(idx) for idx in gesture_sequence]
        # Remove duplicates, keep order
        seen = set()
        filtered = []
        for w, t in words_and_types:
            if w and w not in seen:
                filtered.append((w, t))
                seen.add(w)
        # Sort: SUBJECT, OBJECT, VERB, OTHER
        order = {"SUBJECT": 0, "OBJECT": 1, "VERB": 2, "OTHER": 3}
        filtered.sort(key=lambda x: order.get(x[1], 99))
        sentence = " ".join(w for w, t in filtered)
        return sentence.capitalize() + "."

nlp_processor = TurkishNLPProcessor()

from tensorflow.keras.models import load_model
model = load_model("/Users/dogayalcin/Downloads/bitirmeprojesi/tsl_simple_model_v5.keras")

# MediaPipe setup
mp_holistic = mp.solutions.holistic

def extract_keypoints_from_frame(frame):
    """Extract pose and hand keypoints from a single frame using MediaPipe Holistic."""
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                         results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
        lh = np.array([[res.x, res.y, res.z] for res in
                       results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
        rh = np.array([[res.x, res.y, res.z] for res in
                       results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
        return np.concatenate([pose, lh, rh])

# Streamlit UI Setup
st.title("SignSpeak: Real-Time Turkish Sign Language Recognition")

# Input Selection Sidebar
option = st.sidebar.selectbox("Choose Input", ["Webcam", "Upload Video"])

# Webcam Processing Section
if option == "Webcam":
    st.write("Webcam live feed will appear below:")
    run = st.checkbox("Start Webcam")

    if run:
        # Initialize webcam with specific parameters
        cap = cv2.VideoCapture(0)
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            st.error("Failed to open webcam. Please check your camera permissions.")
        else:
            stframe = st.empty()
            prediction_box = st.empty()
            sentence_box = st.empty()
            timer_box = st.empty()
            sequence = []
            predictions = deque(maxlen=10)
            gesture_sequence = []
            last_prediction = None
            current_sentence = ""
            
            # Timing variables
            collection_time = 15 * 30  # 15 seconds at 30 fps
            display_time = 5 * 30  # 5 seconds at 30 fps
            frame_counter = 0
            phase_start_time = time.time()
            is_collecting = True
            
            try:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to capture frame from webcam.")
                        break

                    # Flip the frame horizontally
                    frame = cv2.flip(frame, 1)
                    
                    # Convert BGR to RGB for display
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Update timing
                    current_time = time.time()
                    elapsed_time = current_time - phase_start_time
                    
                    if is_collecting:
                        # Collection phase (15 seconds)
                        remaining_time = 15 - int(elapsed_time)
                        if remaining_time > 0:
                            timer_box.markdown(f"### Time remaining for gestures: {remaining_time} seconds")
                            
                            # Process frame and make predictions
                            keypoints = extract_keypoints_from_frame(frame)
                            sequence.append(keypoints)
                            if len(sequence) > max_seq_length:
                                sequence.pop(0)

                            if len(sequence) == max_seq_length:
                                input_sequence = pad_sequences([sequence], maxlen=max_seq_length, dtype='float32', padding='post',
                                                            truncating='post')
                                prediction = model.predict(input_sequence)[0]

                                if np.max(prediction) > threshold:
                                    current_prediction = np.argmax(prediction)
                                    predictions.append(current_prediction)

                                    if len(predictions) == predictions.maxlen:
                                        most_common_prediction = max(set(predictions), key=predictions.count)
                                        
                                        if not gesture_sequence or most_common_prediction != gesture_sequence[-1]:
                                            gesture_sequence.append(most_common_prediction)
                                            last_prediction = most_common_prediction
                                            
                                            # Show current word
                                            current_word = CLASS_MAP.get(most_common_prediction, "Unknown")
                                            prediction_box.markdown(f"### Current Sign: ⁠ {current_word} ⁠")
                                            
                                            # Update sentence
                                            current_sentence = nlp_processor.generate_sentence(gesture_sequence)
                                            sentence_box.markdown(f"### Building Sentence: ⁠ {current_sentence} ⁠")
                        else:
                            # Switch to display phase
                            is_collecting = False
                            phase_start_time = current_time
                            if current_sentence:
                                sentence_box.markdown(f"### Final Sentence: ⁠ {current_sentence} ⁠")
                            else:
                                sentence_box.markdown("### No sentence formed")
                    else:
                        # Display phase (5 seconds)
                        remaining_time = 5 - int(elapsed_time)
                        if remaining_time > 0:
                            timer_box.markdown(f"### New collection starts in: {remaining_time} seconds")
                        else:
                            # Reset for next round
                            is_collecting = True
                            phase_start_time = current_time
                            gesture_sequence = []
                            last_prediction = None
                            predictions.clear()
                            current_sentence = ""
                            sentence_box.empty()
                            prediction_box.empty()
                            timer_box.markdown("### Starting new collection phase...")

                    # Display the frame
                    stframe.image(frame_rgb, channels="RGB", use_column_width=True)

            except Exception as e:
                st.error(f"Error during webcam processing: {str(e)}")
            finally:
                cap.release()
    else:
        st.warning("Click the checkbox to start the webcam.")

# Video Upload functionality
elif option == "Upload Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi"])
    if uploaded_video:
        video_path = f"./temp_{uploaded_video.name}"
        with open(video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())

        st.video(video_path)

        st.write("Processing video...")
        cap = cv2.VideoCapture(video_path)
        sequence = []
        predictions = deque(maxlen=5)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Extract keypoints and make predictions
            keypoints = extract_keypoints_from_frame(frame)
            sequence.append(keypoints)
            if len(sequence) > max_seq_length:
                sequence.pop(0)

            if len(sequence) == max_seq_length:
                input_sequence = pad_sequences([sequence], maxlen=max_seq_length, dtype='float32', padding='post',
                                               truncating='post')
                prediction = model.predict(input_sequence)[0]

                if np.max(prediction) > threshold:
                    predictions.append(np.argmax(prediction))

                if len(predictions) == predictions.maxlen:
                    most_common_prediction = max(set(predictions), key=predictions.count)
                    label = CLASS_MAP.get(most_common_prediction, "Unknown")
                    st.write(f"Prediction: {label}")

        cap.release()