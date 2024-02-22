import os
from pathlib import Path
import warnings
import pandas as pd
from mediapipe.python.solutions.holistic import Holistic
from helpers import get_keypoints, insert_keypoints_sequence
from constants import DATA_PATH, FRAME_ACTIONS_PATH, ROOT_PATH

def create_keypoints(frames_folder, save_path):
    """
    Crea keypoints para los frames de una carpeta y los guarda en un archivo HDF5.

    Args:
        frames_folder (str): Ruta de la carpeta que contiene los frames.
        save_path (str): Ruta del archivo HDF5 donde se guardarán los keypoints.
    """
    data = pd.DataFrame([])
    with Holistic() as model_holistic:
        for n_sample, sample_name in enumerate(os.listdir(frames_folder), 1):
            sample_path = os.path.join(frames_folder, sample_name)
            keypoints_sequence = get_keypoints(model_holistic, sample_path)
            data = insert_keypoints_sequence(data, n_sample, keypoints_sequence)
    data.to_hdf(save_path, key="data", mode="w")

def process_folders(words_path):
    """
    Procesa cada carpeta en la ruta dada, generando keypoints para cada una.

    Args:
        words_path (str): Ruta de la carpeta que contiene las palabras.
    """
    for folder_name in os.listdir(words_path):
        folder_path = os.path.join(words_path, folder_name)
        if not os.path.isdir(folder_path):
            continue
        hdf_path = os.path.join(DATA_PATH, f"{folder_name}.h5")
        print(f'Creando keypoints para la carpeta "{folder_name}"...')
        create_keypoints(folder_path, hdf_path)
        print(f"Keypoints creados para la carpeta {folder_name}!")

if __name__ == "__main__":
    words_path = os.path.join(ROOT_PATH, FRAME_ACTIONS_PATH)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        process_folders(words_path)

        #Generar solo de una palabra.

        # word_name = "hola"
        # words_path = os.path.join(words_path, word_name)
        # hdf_path = os.path.join(data_path, f"{word_name}.h5")
        # create_keypoints(words_path, hdf_path)
        # print(f"keypoints creados!")