import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

dataset_path = "shared_data/ppid/Dataset"

X = []
y = []

for class_name in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_path):
        for file in os.listdir(class_path):
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(class_path, file)
                img = Image.open(img_path).convert("L")
                img = img.resize((128, 128))
                X.append(np.array(img).flatten())
                y.append(class_name)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

kernels = ["linear", "poly", "rbf"]
models = {}
accuracies = {}

for k in kernels:
    model = SVC(kernel=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies[k] = acc
    models[k] = model
    print(f"Kernel = {k:<6}: Accuracy = {acc:.2f}")

best_kernel = max(accuracies, key=accuracies.get)
best_model = models[best_kernel]
y_pred_best = best_model.predict(X_test)

plt.figure(figsize=(12, 3))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(X_test[i].reshape(128, 128), cmap="gray")
    color = "green" if y_test[i] == y_pred_best[i] else "red"
    plt.title(f"True: {y_test[i]}\nPred: {y_pred_best[i]}", color=color)
    plt.axis("off")
plt.tight_layout()
plt.show()

cm = confusion_matrix(y_test, y_pred_best, labels=np.unique(y))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
disp.plot(cmap="Blues")
plt.show()
