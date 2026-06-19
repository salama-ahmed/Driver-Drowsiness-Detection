import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import joblib

# ======================
# Load Face & Eye Detectors
# ======================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml')

# ======================
# CLAHE (تحسين الإضاءة)
# ======================
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# ======================
# 1) Load Dataset
# ======================
data = []
labels = []

folders = ["Non Drowsy", "Drowsy"]

print("📂 Loading dataset...")

for label, folder in enumerate(folders):
    path = os.path.join("dataset", folder)

    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # تحسين الإضاءة
        gray = clahe.apply(gray)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # fallback لو مفيش face
        if len(faces) == 0:
            face = gray
        else:
            (x, y, w, h) = faces[0]
            face = gray[y:y+h, x:x+w]

        face = cv2.GaussianBlur(face, (5, 5), 0)

        eyes = eye_cascade.detectMultiScale(face)

        # fallback لو مفيش عين
        if len(eyes) == 0:
            eye = cv2.resize(face, (50, 50))
        else:
            (ex, ey, ew, eh) = eyes[0]
            eye = face[ey:ey+eh, ex:ex+ew]
            eye = cv2.resize(eye, (50, 50))

        # Edge Detection
        edges = cv2.Canny(eye, 50, 150)

        # Data Augmentation
        flipped = cv2.flip(edges, 1)

        data.append(edges.flatten())
        labels.append(label)

        data.append(flipped.flatten())
        labels.append(label)

data = np.array(data)
labels = np.array(labels)

print("Total samples:", len(data))
print("Classes:", np.unique(labels))

# ======================
# 2) Split
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42)

# ======================
# 3) Train or Load Model
# ======================
model_path = "drowsiness_model.pkl"

if os.path.exists(model_path):
    model = joblib.load(model_path)
    print("✅ Model loaded")
else:
    print("🚀 Training started...")
    model = LinearSVC(max_iter=10000)
    model.fit(X_train, y_train)
    print("✅ Training finished!")

    joblib.dump(model, model_path)
    print("💾 Model saved")

# ======================
# 4) Evaluation
# ======================
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# ======================
# 5) Testing
# ======================
print("\n--- Testing ---")

test_folder = "test_images"

if os.path.exists(test_folder):
    for img_name in os.listdir(test_folder):

        img_path = os.path.join(test_folder, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # CLAHE
        gray = clahe.apply(gray)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # fallback
        if len(faces) == 0:
            face = gray
        else:
            (x, y, w, h) = faces[0]
            face = gray[y:y+h, x:x+w]

        face = cv2.GaussianBlur(face, (5, 5), 0)

        eyes = eye_cascade.detectMultiScale(face)

        if len(eyes) == 0:
            eye = cv2.resize(face, (50, 50))
        else:
            (ex, ey, ew, eh) = eyes[0]
            eye = face[ey:ey+eh, ex:ex+ew]
            eye = cv2.resize(eye, (50, 50))

        edges = cv2.Canny(eye, 50, 150)

        test_data = edges.flatten().reshape(1, -1)
        prediction = model.predict(test_data)

        result = "Non Drowsy" if prediction[0] == 0 else "Drowsy"

        print(img_name, "->", result)

        # رسم مستطيل
        if len(faces) != 0:
            (x, y, w, h) = faces[0]
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # كتابة النتيجة
        cv2.putText(img, result, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Result", img)
        cv2.waitKey(2000)

    cv2.destroyAllWindows()

else:
    print("❌ test_images folder not found")
