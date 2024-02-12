import pandas as pd
import tensorflow as tf
from sklearn.metrics import binary_accuracy
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K

# Load preprocessed data
test_data = pd.read_csv('test_data.csv')

# Load the trained model
model = tf.keras.models.load_model('video_recommendation_model.h5')

# Make predictions
predictions, item_embeddings = model.predict([test_data['user_id'], test_data['video_id']])

# Binary cross-entropy loss
prediction_loss = binary_crossentropy(test_data['engagement'], predictions)
print(f'Binary Cross-Entropy Loss on Test Set: {K.eval(K.mean(prediction_loss))}')

# Cosine similarity loss
true_embeddings = model.layers[4].get_weights()[0][test_data['user_id']]
similarity_loss = 1 - K.sum(true_embeddings * item_embeddings, axis=-1)
print(f'Cosine Similarity Loss on Test Set: {K.eval(K.mean(similarity_loss))}')

# Calculate binary accuracy
binary_acc = binary_accuracy(test_data['engagement'], predictions)
print(f'Binary Accuracy on Test Set: {K.eval(K.mean(binary_acc))}')
