import tensorflow as tf
model=tf.keras.models.load_model("DBA_COLAB.hdf5")
converter = tf.lite.TFLiteConverter.from_keras_model(model) # TF 2.x
tflite_model = converter.convert()
tflite_model_file_name = "DBA_" + tf.__version__ + "_" + str(date.today()) + ".tflite"
open(tflite_model_file_name, "wb").write(tflite_model)
