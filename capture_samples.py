import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from helpers import create_folder, draw_keypoints, mediapipe_detection, save_frames, there_hand
from constants import FONT, FONT_POS, FONT_SIZE, FRAME_ACTIONS_PATH, ROOT_PATH
from pytube import YouTube

def download_youtube_video(youtube_url, output_path):
    yt = YouTube(youtube_url)
    yt.streams.filter(progressive=True, file_extension='mp4').first().download(output_path)

def capture_hand_frames_from_youtube(youtube_url, output_folder):
    create_folder(output_folder)
    print(f'Carpeta de salida creada en: {output_folder}')  # Mensaje de depuración

    # Descargar el video de YouTube
    download_youtube_video(youtube_url, output_folder)
    print("Video descargado exitosamente")

    video_filename = os.path.join(output_folder, os.listdir(output_folder)[0])

    print("Comenzando la captura de frames con manos detectadas...")  # Mensaje de depuración

    with Holistic() as holistic_model:
        cap = cv2.VideoCapture(video_filename)
        if not cap.isOpened():
            print("Error: No se pudo abrir el video descargado")
            return

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image, results = mediapipe_detection(frame, holistic_model)

            if there_hand(results):
                frame_filename = os.path.join(output_folder, f"hand_frame_{frame_count}.jpg")
                cv2.imwrite(frame_filename, frame)
                print(f"Frame con manos guardado en: {frame_filename}")  # Mensaje de depuración
                frame_count += 1

        cap.release()
        cv2.destroyAllWindows()

        # Eliminar el archivo de video después de la captura de frames
        os.remove(video_filename)
        print("Archivo de video eliminado")  # Mensaje de depuración

    print("Fin de la captura de frames con manos detectadas")  # Mensaje de depuración

if __name__ == "__main__":
    youtube_url = input("Por favor, introduce la URL del video de YouTube: ")
    output_folder_name = input("Por favor, introduce el nombre de la carpeta para guardar los frames: ")
    output_folder = os.path.join(ROOT_PATH, FRAME_ACTIONS_PATH, output_folder_name)
    capture_hand_frames_from_youtube(youtube_url, output_folder)
