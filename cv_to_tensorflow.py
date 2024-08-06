import tensorflow as tf

# Load the model from .h5 file
model = tf.keras.models.load_model('keras/test_4.h5')

# Save the model in SavedModel format
model.export('model_tfjs')
