import os
from capture_samples import capture_samples
from create_keypoints import create_keypoints

root = os.getcwd()
words_path = os.path.join(root, "action_frames")
data_path = os.path.join(root, "data")

#Capturar muestras para una palabra
# word_name = "censurado!"
# words_path = os.path.join(words_path, word_name)
# capture_samples(words_path)

#Generar Keypoints de todas las palabras.
# for word_name in os.listdir(words_path):
#     words_path = os.path.join(words_path, word_name)
#     hdf_path = os.path.join(data_path, f"{word_name}.h5")
#     print(f'Creando keypoints de "{word_name}"...')
#     create_keypoints(words_path, hdf_path)