import tensorflow as tf

# Define the path where the model weights will be saved
vgg_weights_path = "Art style transfer project/VGG19weights/VGG19.weights.h5"

# Download the weights if not already present
vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
vgg.save_weights(vgg_weights_path)

print("VGG19 weights downloaded and saved locally.")
