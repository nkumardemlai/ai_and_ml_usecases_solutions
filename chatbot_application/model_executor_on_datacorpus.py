import os
import json
import pickle
import random 
import numpy as np 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Embedding, LSTM, Dense

import tensorflow as tf
print("TensorFlow version:", tf.__version__)
import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))


# Directory path containing the .txt files
directory = r'C:\Users\pedam\github_checkin_code\ml_learning_notebooks\src_data\corpus_data'
intents = r'C:\Users\pedam\github_checkin_code\ml_learning_notebooks\src_data\corpus_data\intents.json'

# Generator function to yield chunks of content from files along with their labels
def read_chunks_with_labels(directory, chunk_size=4096):
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            label = 'movie' if 'movie' in filename else 'politics'
            with open(file_path, 'r') as file:
                while True:
                    chunk = file.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk, label


# Combine content of all .txt files along with their labels
combined_data = []
for chunk, label in read_chunks_with_labels(directory):
    combined_data.append((chunk, label))

# Print 5 random lines from the combined data
random_samples = random.sample(combined_data, 2)
for chunk, label in random_samples:
    print(f"Label: {label}")
    print(f"Chunk: {chunk}")

with open(intents) as file:
    data = json.load(file)
    
training_sentences = []
training_labels = []
labels = []
responses = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        training_sentences.append(pattern)
        training_labels.append(intent['tag'])
    responses.append(intent['responses'])
    
    if intent['tag'] not in labels:
        labels.append(intent['tag'])
        
num_classes = len(labels)


lbl_encoder = LabelEncoder()
lbl_encoder.fit(training_labels)
training_labels = lbl_encoder.transform(training_labels)

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

movie_list = []
politics_list = []

# Segregate data based on labels
for data, label in combined_data:
    if label == 'movie':
        movie_list.append(data)
    elif label == 'politics':
        politics_list.append(data)

# Get GPU device name
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    device_name = gpu_devices[0].name
    print("GPU device name:", device_name)
else:
    print("No GPU available.")


if gpu_devices:
    # Only use the first GPU device
    try:
        tf.config.experimental.set_visible_devices(gpu_devices[0], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpu_devices), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        print(e)


 # Combine data and create labels
all_data = movie_list + politics_list
labels = ['movie'] * len(movie_list) + ['politics'] * len(politics_list)
# Tokenize text data
tokenizer = Tokenizer() 
tokenizer.fit_on_texts(all_data)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(all_data)
# Pad sequences
max_sequence_length = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')
# Convert labels to numerical format
label_dict = {'movie': 0, 'politics': 1}
labels_numeric = np.array([label_dict[label] for label in labels])
# Define model
model = Sequential([
     Embedding(len(word_index) + 1, 128, input_shape=(max_sequence_length,)),
     LSTM(128),
     Dense(64, activation='relu'),
     Dense(2, activation='softmax')  # 2 classes: movie and politics
])
# Compile model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Train model
model.fit(padded_sequences, labels_numeric, epochs=10, batch_size=32)
# Save model and tokenizer for later use
model.save('intent_classifier_model.h5')
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# to save the fitted label encoder
with open('label_encoder.pickle', 'wb') as ecn_file:
    pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)
    
model.summary()



        
