import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K

def cosine_similarity_loss(y_true, y_pred):
    # Normalize the embeddings to unit length
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)

    # Calculate cosine similarity
    similarity = K.sum(y_true * y_pred, axis=-1)

    # Cosine similarity loss: maximize similarity between embeddings
    return 1 - similarity

def combined_loss(alpha=0.5):
    # Weighted combination of binary cross-entropy and cosine similarity losses
    def loss(y_true, y_pred):
        prediction_loss = binary_crossentropy(y_true, y_pred)
        similarity_loss = cosine_similarity_loss(y_true, y_pred)
        return alpha * prediction_loss + (1 - alpha) * similarity_loss

    return loss

# Load preprocessed data
train_data = pd.read_csv('train_data.csv')

# Define the model
num_users = len(train_data['user_id'].unique())
num_videos = len(train_data['video_id'].unique())
embedding_size = 30

user_input = Input(shape=(1,))
video_input = Input(shape=(1,))

user_embedding_layer = Embedding(input_dim=num_users, output_dim=embedding_size, input_length=1)
video_embedding_layer = Embedding(input_dim=num_videos, output_dim=embedding_size, input_length=1)

user_embedding = user_embedding_layer(user_input)
video_embedding = video_embedding_layer(video_input)

user_embedding = Flatten()(user_embedding)
video_embedding = Flatten()(video_embedding)

concatenated = Concatenate()([user_embedding, video_embedding])

dense_layer_1 = Dense(64, activation='relu')(concatenated)
output_layer = Dense(1, activation='linear')(dense_layer_1)

model = Model(inputs=[user_input, video_input], outputs=output_layer)
model.compile(optimizer=Adam(learning_rate=0.001), loss=combined_loss(alpha=0.5))

# Train the model
model.fit([train_data['user_id'], train_data['video_id']], train_data['engagement'], epochs=10, batch_size=32, validation_split=0.2)

# Save the trained model
model.save('video_recommendation_model.h5')
