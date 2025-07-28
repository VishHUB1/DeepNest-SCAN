import tensorflow as tf

print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("TensorFlow is using the GPU")
    print(gpus)
else:
    print("TensorFlow is NOT using the GPU")

print(tf.__version__)

