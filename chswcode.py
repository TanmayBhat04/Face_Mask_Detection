import os
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report

# Load the TensorFlow Lite model
model_path = "/kaggle/input/my_latest_model/tensorflow2/default/1/trained.tflite"  # Update with your model path
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class index mapping (edit as per Edge Impulse class order if different)
class_names = ["cloth", "n95", "no_face_mask"]

# Preprocess images to match model requirements
def preprocess_image(image_path, input_details):
    img = Image.open(image_path).resize((96, 96))  # Match input size
    img_array = np.array(img) / 255.0  # Normalize

    if input_details[0]['dtype'] == np.int8:
        scale, zero_point = input_details[0]['quantization']
        img_array = img_array / scale + zero_point
        img_array = np.clip(img_array, -128, 127).astype(np.int8)
    else:
        img_array = img_array.astype(np.float32)

    return np.expand_dims(img_array, axis=0)

# Predict the class for a single image
def predict_image(image_path, input_details, interpreter):
    input_data = preprocess_image(image_path, input_details)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_index = int(np.argmax(output_data[0]))
    confidence = output_data[0][predicted_index]

    predicted_class = class_names[predicted_index]

    # Format prediction
    if predicted_class == "no_face_mask":
        label = "no_face_mask"
    elif predicted_class in ["cloth", "n95"]:
        label = f"masked ({predicted_class})"
    else:
        label = "masked (other)"

    return label, predicted_index, confidence

# Evaluate a folder
def evaluate_folder(folder_path, actual_class, input_details, interpreter):
    correct = 0
    total = 0
    y_true = []
    y_pred = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            total += 1
            image_path = os.path.join(folder_path, filename)
            label, predicted_index, _ = predict_image(image_path, input_details, interpreter)

            true_index = class_names.index(actual_class)
            y_true.append(true_index)
            y_pred.append(predicted_index)

            if predicted_index == true_index:
                correct += 1

    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"Folder: {folder_path} | Accuracy: {accuracy:.2f}% ({correct}/{total})")
    return y_true, y_pred

# Define folders and evaluate
cloth_folder = "/kaggle/input/cloth-dataset4"
n95_folder = "/kaggle/input/n95-dataset4"
nfm_folder = "/kaggle/input/nfm-dataset"  # Adjust to your folder name

y_true_all, y_pred_all = [], []

for folder_path, actual_class in [
    (cloth_folder, "cloth"),
    (n95_folder, "n95"),
    (nfm_folder, "no_face_mask")
]:
    y_true, y_pred = evaluate_folder(folder_path, actual_class, input_details, interpreter)
    y_true_all.extend(y_true)
    y_pred_all.extend(y_pred)

# Print final metrics
from collections import Counter

print("\nConfusion Matrix:")
print(confusion_matrix(y_true_all, y_pred_all))

print("\nClassification Report:")
print(classification_report(y_true_all, y_pred_all, target_names=class_names))

# Calculate weighted average accuracy
correct_counts = np.array(confusion_matrix(y_true_all, y_pred_all)).diagonal()
class_counts = Counter(y_true_all)
total_samples = sum(class_counts.values())

weighted_accuracy = sum(correct_counts[i] for i in range(len(class_names))) / total_samples * 100
print(f"\n Overall Accuracy: {weighted_accuracy:.2f}%")
