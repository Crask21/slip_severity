#Following code shows the use of the five classifiers for slip detection.
import csv
import numpy as np
import joblib
import os
#jump one directory up to access the model files
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(root_path, "models", "detection_rf_ff.pkl")
detection_classifier = joblib.load(model_path)


data_path = os.path.join(root_path, "data", "non_contact.csv")
import pandas as pd
test_data = pd.read_csv(data_path)


fingers = []
for i in range(5):
    finger_data = test_data.iloc[0][1+i*51:1+(i+1)*51].values.astype(float)
    fingers.append(finger_data)
fingers = np.array(fingers)

for i in range(fingers.shape[0]):
    finger_data = fingers[i].reshape(1, -1)
    prediction = detection_classifier.predict(finger_data)
    print(f"Finger {i+1} prediction: {prediction[0]}")

