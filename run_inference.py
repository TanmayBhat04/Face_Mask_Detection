import os
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report

# Load the TensorFlow Lite model
model_path = "model/trained.tflite"  # Updated path for local or Docker use
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Preprocess images to match model requirements
def preprocess_image(image_path, input_details):
    img = Image.open(image_path).resize((96, 96))  # Resize to model's input size
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]

    # Check model input type and process accordingly
    if input_details[0]['dtype'] == np.int8:
        scale, zero_point = input_details[0]['quantization']
        img_array = img_array / scale + zero_point
        img_array = np.clip(img_array, -128, 127).astype(np.int8)
    else:
        img_array = img_array.astype(np.float32)

    return np.expand_dims(img_array, axis=0)

# Predict the class for a single image
def predict_image(image_path, input_details, interpreter, threshold=0.5):
    input_data = preprocess_image(image_path, input_details)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Adjust the threshold for "Mask Detected" prediction
    predicted_class = 0 if output_data[0][0] > output_data[0][1] and output_data[0][0] > threshold else 1
    return predicted_class, output_data

# Evaluate predictions for a folder
def evaluate_folder(folder_path, label, input_details, interpreter, threshold=0.5):
    correct_predictions = 0
    total_images = 0
    y_true = []
    y_pred = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            total_images += 1
            image_path = os.path.join(folder_path, filename)
            predicted_class, output_data = predict_image(image_path, input_details, interpreter, threshold)
            
            # Append for confusion matrix
            y_true.append(label)
            y_pred.append(predicted_class)

            # Check if the prediction matches the actual label
            if predicted_class == label:
                correct_predictions += 1

    accuracy = (correct_predictions / total_images) * 100 if total_images > 0 else 0
    print(f"Folder: {folder_path}, Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_images})")

    return correct_predictions, total_images, y_true, y_pred

# Paths to image folders (updated for Docker/local)
masked_folder = "test_images/masked"
non_masked_folder = "test_images/non_masked"

# Evaluate both folders
masked_correct, masked_total, masked_y_true, masked_y_pred = evaluate_folder(
    masked_folder, label=0, input_details=input_details, interpreter=interpreter, threshold=0.5
)
non_masked_correct, non_masked_total, non_masked_y_true, non_masked_y_pred = evaluate_folder(
    non_masked_folder, label=1, input_details=input_details, interpreter=interpreter, threshold=0.5
)

# Combine results for overall metrics
y_true = masked_y_true + non_masked_y_true
y_pred = masked_y_pred + non_masked_y_pred

# Calculate overall accuracy
overall_correct = masked_correct + non_masked_correct
overall_total = masked_total + non_masked_total
overall_accuracy = (overall_correct / overall_total) * 100 if overall_total > 0 else 0

print(f"\nOverall Accuracy: {overall_accuracy:.2f}% ({overall_correct}/{overall_total})")

# Confusion matrix and classification report
conf_matrix = confusion_matrix(y_true, y_pred)
class_report = classification_report(y_true, y_pred, target_names=["MASKED", "NON-MASKED"])

print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)
