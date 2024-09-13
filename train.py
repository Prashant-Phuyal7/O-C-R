import tensorflow as tf
try: [tf.config.experimental.set_memory_growth(gpu, True) for gpu in tf.config.experimental.list_physical_devices("GPU")]
except: pass

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard # type: ignore

from mltu.preprocessors import ImageReader
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding, ImageShowCV2
from mltu.augmentors import RandomBrightness, RandomRotate, RandomErodeDilate, RandomSharpen
from mltu.annotations.images import CVImage
from mltu.tensorflow.dataProvider import DataProvider
from mltu.tensorflow.losses import CTCloss
from mltu.tensorflow.callbacks import Model2onnx, TrainLogger
from mltu.tensorflow.metrics import CWERMetric

from model import train_model
from configs import ModelConfigs

import os
import tarfile
import cv2
import glob
import numpy as np
from tqdm import tqdm
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile

dataset_path = r'Datasets\IAM_Words\words\a01'
grayscale_folder_path = r'grey_scaledimages'
os.makedirs(grayscale_folder_path, exist_ok=True)
word_filepath = r'Datasets'
output_file = os.path.join(word_filepath, "words.txt")
os.makedirs(word_filepath, exist_ok=True)


image_files = glob.glob(os.path.join(dataset_path, '*.png'))  
if not image_files:
    raise FileNotFoundError(f"No image files found in the dataset path: {dataset_path}")

with open(output_file, 'w') as f:
    f.write("#--- words.txt ---------------------------------------------------------------#\n")
    f.write("# iam database word information\n")
    f.write("# format: word_id segmentation graylevel num_components x y w h tag transcription\n")
    f.write("#\n")

    # Processing original images and saving grayscale versions
    for image_file in tqdm(image_files, desc="Processing original images"):
        grayscale_image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

        if grayscale_image is None:
            print(f"Error reading file: {image_file}")
            continue

        # Save the grayscale image to the specified folder
        grayscale_image_path = os.path.join(grayscale_folder_path, os.path.basename(image_file))
        if not cv2.imwrite(grayscale_image_path, grayscale_image):
            print(f"Error saving file: {grayscale_image_path}")
            continue

    # saving all grayscale images and processing them from the grayscale folder
    grayscale_image_files = glob.glob(os.path.join(grayscale_folder_path, '*.png'))

    if not grayscale_image_files:
        raise FileNotFoundError(f"No grayscale images found in the specified directory: {grayscale_folder_path}")

    for grayscale_image_path in tqdm(grayscale_image_files, desc="Processing grayscale images"):
        grayscale_image = cv2.imread(grayscale_image_path, cv2.IMREAD_GRAYSCALE)

        if grayscale_image is None:
            print(f"Error reading file: {grayscale_image_path}")
            continue

        # Applying Otsu's thresholding to find a suitable threshold value
        graylevel_threshold, binary_image = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        print(f"Type of graylevel_threshold: {type(graylevel_threshold)}")
        print(f"Value of graylevel_threshold: {graylevel_threshold}")

        graylevel_threshold = int(graylevel_threshold)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image)

        highest_components = 0
        highest_components_details = ""

        for i in range(1, num_labels):
            x, y, width, height, area = stats[i]

            word_image = binary_image[y:y+height, x:x+width]
            num_word_components, _, _, _ = cv2.connectedComponentsWithStats(word_image)

            word_id = f"{os.path.splitext(os.path.basename(grayscale_image_path))[0]}-{i:03d}"
            segmentation_result = "ok"
            tag = "NN"
            transcription = "?"

           
            print(f"Processed word: {word_id} with {num_word_components - 1} components")
            num_components = num_word_components - 1  

            
        if num_components > highest_components:
            highest_components = num_components
            highest_components_details = (
                f"{word_id} {segmentation_result} {graylevel_threshold} {highest_components} "
                f"{x} {y} {width} {height} {tag} {transcription}\n"
            )
            print(f"New highest: {word_id} with {highest_components} components")
        if highest_components_details:
            f.write(highest_components_details)
            print(f"Wrote highest component details: {highest_components_details.strip()}")






# Initialize dataset variables
dataset, vocab, max_len = [], set(), 0

# Path to the words.txt file
words_file_path = os.path.join(word_filepath, "words.txt")
words = open(words_file_path, "r").readlines()

# List to keep track of missing files
missing_files = []

# Processing lines from words.txt
for line in tqdm(words, desc="Processing words"):
    # Skip comment lines
    if line.startswith("#"):
        continue

    # Split the line by spaces to extract components
    line_split = line.strip().split(" ")

    # Skip lines marked with 'err'
    if line_split[1] == "err":
        continue

    # Extract folder and filename information
    folder1 = line_split[0][:3]  # Extracts the folder name, e.g., 'a01'
    file_name = line_split[0] + ".png"  # Constructs the image file name, e.g., 'a01-000u-00.png'
    label = line_split[-1]  

    # Construct the absolute path to the image file
    rel_path = os.path.join(dataset_path, "..", folder1, file_name)  # Considering the base path is 'words\a01'
    abs_path = os.path.abspath(rel_path)

    # Check if the file exists at the constructed path
    if not os.path.exists(abs_path):
        print(f"File not found: {abs_path}")
        missing_files.append(abs_path)  # Log missing file
        continue

    # Append the path and label to the dataset list
    dataset.append([abs_path, label])
    vocab.update(list(label))
    max_len = max(max_len, len(label))

print("Dataset processing complete.")

# Save missing files to a log file for debugging
if missing_files:
    with open("missing_files_log.txt", "w") as log_file:
        for missing_file in missing_files:
            log_file.write(f"{missing_file}\n")
    print(f"{len(missing_files)} files were not found. Details are saved in 'missing_files_log.txt'.")




# Create a ModelConfigs object to store model configurations
configs = ModelConfigs()

# Save vocab and maximum text length to configs
configs.vocab = "".join(vocab)
configs.max_text_length = max_len
configs.save()

# Create a data provider for the dataset
data_provider = DataProvider(
    dataset=dataset,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[ImageReader(CVImage)],
    transformers=[
        ImageResizer(configs.width, configs.height, keep_aspect_ratio=False),
        LabelIndexer(configs.vocab),
        LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab)),
    ],
)

# Split the dataset into training and validation sets
train_data_provider, val_data_provider = data_provider.split(split=0.9)

# Augment training data with random brightness, rotation, and erode/dilate
train_data_provider.augmentors = [
    RandomBrightness(),
    RandomErodeDilate(),
    RandomSharpen(),
    RandomRotate(angle=10),
]

# Creating TensorFlow model architecture
model = train_model(
    input_dim=(configs.height, configs.width, 3),
    output_dim=len(configs.vocab),
)

# Compile the model and print summary
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=configs.learning_rate),
    loss=CTCloss(),
    metrics=[CWERMetric(padding_token=len(configs.vocab))],
)
model.summary(line_length=110)

# Define callbacks
configs.model_path = configs.model_path + ".keras"
earlystopper = EarlyStopping(monitor="val_CER", patience=20, verbose=1)
checkpoint = ModelCheckpoint(f"{configs.model_path}", monitor="val_CER", verbose=1, save_best_only=True, mode="min")
trainLogger = TrainLogger(configs.model_path)
tb_callback = TensorBoard(f"{configs.model_path}/logs", update_freq=1)
reduceLROnPlat = ReduceLROnPlateau(monitor="val_CER", factor=0.9, min_delta=1e-10, patience=10, verbose=1, mode="auto")
model2onnx = Model2onnx(f"{configs.model_path}")

# Train the model
model.fit(
    train_data_provider,
    validation_data=val_data_provider,
    epochs=configs.train_epochs,
    callbacks=[earlystopper, checkpoint, trainLogger, reduceLROnPlat, tb_callback, model2onnx],
    # workers=configs.train_workers
)


# Save training and validation datasets as CSV files
train_data_provider.to_csv(os.path.join(configs.model_path, "train.csv"))
val_data_provider.to_csv(os.path.join(configs.model_path, "val.csv"))