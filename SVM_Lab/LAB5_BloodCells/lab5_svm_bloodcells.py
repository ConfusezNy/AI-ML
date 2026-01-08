import os
import random
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

dataset_path = "shared_data/bloodcells/bloodcells_dataset"
max_images_per_class = 120

X = []
y = []

for class_name in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_path):
        files = [f for f in os.listdir(class_path) if f.lower().endswith((".jpg",".png",".jpeg"))]
        random.shuffle(files)
        files = files[:max_images_per_class]

        for file in files:
            img_path = os.path.join(class_path, file)
            img = Image.open(img_path).convert("L")
            img = img.resize((32, 32))
            X.append(np.array(img).flatten())
            y.append(class_name)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "linear": LinearSVC(max_iter=3000),
    "poly": SVC(kernel="poly", degree=3, max_iter=2000),
    "rbf": SVC(kernel="rbf", max_iter=2000)
}

accuracies = {}
predictions = {}

for k, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies[k] = acc
    predictions[k] = y_pred
    print(f"Kernel = {k:<6}: Accuracy = {acc:.2f}")

best_kernel = max(accuracies, key=accuracies.get)
y_pred_best = predictions[best_kernel]

plt.figure(figsize=(12, 3))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(X_test[i].reshape(32, 32), cmap="gray")
    color = "green" if y_test[i] == y_pred_best[i] else "red"
    plt.title(f"True: {y_test[i]}\nPred: {y_pred_best[i]}", color=color)
    plt.axis("off")
plt.tight_layout()
plt.show()

cm = confusion_matrix(y_test, y_pred_best, labels=np.unique(y))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
disp.plot(cmap="Blues")
plt.show()
