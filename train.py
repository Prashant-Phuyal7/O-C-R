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
from tqdm import tqdm
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile


dataset_path = 'Datasets/IAM_Words'
grayscale_dataset_path = os.path.join("Datasets", "IAM_Words_Grayscale")

os.makedirs(grayscale_dataset_path, exist_ok=True)
output_file_path = os.path.join("Datasets", "IAM_Words", "words.txt")

lines = []

# Traverse the dataset directory
for root, dirs, files in os.walk(dataset_path):
    for file in tqdm(files):
        if file.endswith(".png"):  # Check for PNG image files
            # Construct the full path of the image file
            image_path = os.path.join(root, file)
            
            # Extract the identifier from the folder structure
            path_parts = image_path.split(os.sep)  # Split path into components
            if len(path_parts) < 3:
                print(f"Skipping invalid path: {image_path}")
                continue

            folder1 = path_parts[-3]  # Get the first folder part (e.g., 'a01')
            folder2 = path_parts[-2]  # Get the second folder part (e.g., 'a01-000u')
            file_id = os.path.splitext(file)[0]  # Extract the file name without extension (e.g., 'a01-000u-00-00')

            # Correct identifier format
            identifier = f"{folder1}-{file_id}"

            # Extract the actual data needed from dataset annotations
            # Replace these placeholders with actual data extraction logic
            gray_level = 154  # Example gray level; replace with actual data
            num_components = 1  # Number of components; replace with actual data
            bounding_box = "408 768 27 51"  # Replace with actual bounding box data (x y w h)
            grammatical_tag = "AT"  # Replace with actual tag
            transcription = "A"  # Replace with actual transcription

            # Construct the line for words.txt
            line = f"{identifier} ok {gray_level} {num_components} {bounding_box} {grammatical_tag} {transcription}\n"
            
            # Add the line to the list
            lines.append(line)

# Write all the collected lines to the words.txt file
with open(output_file_path, "w") as f:
    f.writelines(lines)

print(f"'words.txt' has been created successfully at: {output_file_path}")


if not os.path.exists(dataset_path):
    file = tarfile.open(os.path.join(dataset_path, "words.tgz"))
    file.extractall(os.path.join(dataset_path, "words"))

dataset, vocab, max_len = [], set(), 0

words = open(os.path.join(dataset_path, "words.txt"), "r").readlines()

for line in tqdm(words):
    if line.startswith("#"):
        continue

    line_split = line.split(" ")
    if line_split[1] == "err":
        continue

    folder1 = line_split[0][:3]
    file_name = line_split[0] + ".png"
    label = line_split[-1].rstrip("\n")

    rel_path = os.path.join(dataset_path, "words", folder1, file_name)
    if not os.path.exists(rel_path):
        print(f"File not found: {rel_path}")
        continue

    original_path = os.path.join(dataset_path, "words", folder1, file_name)
    if not os.path.exists(original_path):
        print(f"File not found: {original_path}")
        continue

    dataset.append([rel_path, label])
    vocab.update(list(label))
    max_len = max(max_len, len(label))



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
train_data_provider, val_data_provider = data_provider.split(split = 0.9)

# Augment training data with random brightness, rotation and erode/dilate
train_data_provider.augmentors = [
    RandomBrightness(), 
    RandomErodeDilate(),
    RandomSharpen(),
    RandomRotate(angle=10), 
    ]

# Creating TensorFlow model architecture
model = train_model(
    input_dim = (configs.height, configs.width, 3),
    output_dim = len(configs.vocab),
)

# Compile the model and print summary
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=configs.learning_rate), 
    loss=CTCloss(), 
    metrics=[CWERMetric(padding_token=len(configs.vocab))],
)
model.summary(line_length=110)

# Define callbacks
configs.model_path=configs.model_path+".keras"
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

# Save training and validation datasets as csv files
train_data_provider.to_csv(os.path.join(configs.model_path, "train.csv"))
val_data_provider.to_csv(os.path.join(configs.model_path, "val.csv"))