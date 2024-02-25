import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, render_template, Response

import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from keras.models import load_model
from helpers import draw_keypoints, extract_keypoints, format_sentences, get_actions, mediapipe_detection, save_txt, there_hand
from text_to_speech import text_to_speech
from constants import DATA_PATH, FONT, FONT_POS, FONT_SIZE, MAX_LENGTH_FRAMES, MIN_LENGHT_FRAMES, MODELS_PATH, MODEL_NAME, ROOT_PATH
import threading

app = Flask(__name__)

def evaluate_model(model):
    count_frame = 0
    repe_sent = 1
    kp_sequence, sentence = [], []
    actions = get_actions(DATA_PATH)

    with Holistic() as holistic_model:
        video = cv2.VideoCapture(0)

        while video.isOpened():
            _, frame = video.read()

            image, results = mediapipe_detection(frame, holistic_model)
            kp_sequence.append(extract_keypoints(results))

            if len(kp_sequence) > MAX_LENGTH_FRAMES and there_hand(results):
                count_frame += 1

            else:
                if count_frame >= MIN_LENGHT_FRAMES:
                    res = model.predict(np.expand_dims(kp_sequence[-MAX_LENGTH_FRAMES:], axis=0))[0]

                    if res[np.argmax(res)] > 0.7:  # Umbral de confianza
                        sent = actions[np.argmax(res)]
                        sentence.insert(0, sent)
                        threading.Thread(target=text_to_speech, args=(sent,)).start()
                        sentence, repe_sent = format_sentences(sent, sentence, repe_sent)

                    count_frame = 0
                    kp_sequence = []

            cv2.rectangle(image, (0,0), (640, 35), (245, 117, 16), -1)
            cv2.putText(image, ' | '.join(sentence), FONT_POS, FONT, FONT_SIZE, (255, 255, 255))
            save_txt('outputs/sentences.txt', '\n'.join(sentence))

            draw_keypoints(image, results)

            _, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    model_path = os.path.join(MODELS_PATH, MODEL_NAME)  # Coloca aquí la ruta de tu modelo
    lstm_model = load_model(model_path)
    return Response(evaluate_model(lstm_model), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
