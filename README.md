# Week 2 Workshop: Hand Gesture Recognition Using Machine Learning

## 1. Project Overview

Welcome to Week 2 of the Machine Learning workshop!

In this project, you will build a machine learning model that predicts the number of fingers shown in a hand image.

The model will classify images into six classes:

| Label | Gesture |
|---|---|
| 0 | Zero fingers / closed hand |
| 1 | One finger |
| 2 | Two fingers |
| 3 | Three fingers |
| 4 | Four fingers |
| 5 | Five fingers |

You will use Python, Google Colab, OpenCV, NumPy, Matplotlib, Scikit-learn, and Joblib.

**Your goal:** Understand the complete machine learning workflow, from loading an image dataset to training, testing, evaluating, and saving a model.

> Important: This is a beginner-level machine learning project. The model's performance depends on the quality and variety of your dataset. It may not recognize every hand gesture correctly under different lighting conditions, backgrounds, or camera angles.

---

## 2. Start Here: Google Colab Notebook

Open the notebook provided for this workshop:

**[Open the Hand Gesture Recognition Notebook in Google Colab](https://colab.research.google.com/drive/1vzMOrhKLZDNOd7mTQjCb2rYZhccwKdmb?usp=sharing)**

### Instructions

1. Open the notebook using the link above.
2. If you cannot edit the original notebook, select **File → Save a copy in Drive**.
3. Read the explanation provided before each code cell.
4. Run the cells in order, from top to bottom.
5. Check the output after every important step.
6. Troubleshoot any errors before moving forward.
7. Save your completed notebook and submit it along with your GitHub repository.

**Do not simply run the code without understanding it.** You should be able to explain what each step does, why it is necessary, and how it contributes to the final model.

---

## 3. Learning Objectives

By completing this project, you should be able to:

1. Explain the difference between traditional programming and machine learning.
2. Understand supervised learning and image classification.
3. Organize an image dataset using class labels.
4. Load and visualize images using OpenCV.
5. Resize and normalize images.
6. Convert images into numerical features.
7. Divide a dataset into training and testing subsets.
8. Train a Support Vector Machine (SVM) classifier.
9. Predict labels for images.
10. Evaluate a model using accuracy and a confusion matrix.
11. Save a trained model.
12. Document your work using GitHub and a README file.

---

## 4. How the Project Works

Your machine learning workflow follows these stages:

```text
Image Dataset
     |
     v
Load and Inspect Images
     |
     v
Preprocess Images
(Resize, Flatten, Normalize)
     |
     v
Create Features and Labels
     |
     v
Split into Training and Testing Data
     |
     v
Train the SVM Model
     |
     v
Predict Gesture Labels
     |
     v
Evaluate Accuracy and Confusion Matrix
     |
     v
Test a New Image
     |
     v
Save the Trained Model
```

### Why do we need this pipeline?

A computer does not automatically understand an image as a human does. We must represent the image numerically, provide labeled examples, train a model, and test whether it can make useful predictions on data it has not seen during training.

---

## 5. Tools and Technologies

| Tool | Purpose |
|---|---|
| Python | Main programming language |
| Google Colab | Write and execute Python code in a browser |
| Git | Track changes to project files |
| GitHub | Store and share your project |
| OpenCV | Load, resize, and process images |
| NumPy | Store and manipulate numerical arrays |
| Matplotlib | Display images and evaluation results |
| Scikit-learn | Split data, train the model, and evaluate predictions |
| Joblib | Save the trained model |

---

## 6. Step 1 — Create Your GitHub Repository

### What to do

Create a GitHub repository named:

`hand-gesture-recognition`

### Why do this?

A repository stores your source code, documentation, and project files in one place. It also lets your instructor review your work.

### How to do it

1. Open https://github.com.
2. Sign in to your account.
3. Click **New repository**.
4. Enter the repository name: `hand-gesture-recognition`.
5. Add the description: `Hand gesture classification using machine learning`.
6. Choose Public or Private according to your instructor's instructions.
7. Select **Add a README file**.
8. Click **Create repository**.

Your repository should eventually contain files similar to these:

```text
hand-gesture-recognition/
├── README.md
├── hand_gesture_dataset.zip
├── hand_gesture_recognition.ipynb
├── requirements.txt
└── .gitignore
```

The ZIP filename above is an example. Replace it with the actual filename of the dataset you upload.

---

## 7. Step 2 — Download the Dataset ZIP from GitHub

The hand gesture image dataset is provided as a ZIP file in this repository.

### What to do

Download the dataset ZIP file from GitHub before running the notebook.

### Why is this necessary?

The model learns from example images. Each image must have a correct class label indicating the number of fingers shown.

### How to do it

1. Open your GitHub repository.
2. Locate the dataset ZIP file.
3. Click the file and download it to your computer.
4. Keep track of the downloaded filename.
5. Open the provided Google Colab notebook.
6. Upload the ZIP file into the Colab session using the next step.

**Important:** Uploading a ZIP file to GitHub does not automatically make it available inside Colab. You must download and upload it, or implement another explicit method to access it.

---

## 8. Step 3 — Open Google Colab

### What to do

Open the provided notebook:

**[Open the workshop notebook](https://colab.research.google.com/drive/1vzMOrhKLZDNOd7mTQjCb2rYZhccwKdmb?usp=sharing)**

### Instructions

1. Open the notebook.
2. Save a personal copy in Google Drive if needed.
3. Read each code cell.
4. Run the cells in sequence.
5. Check the output after each step.

### Why use Google Colab?

Google Colab allows you to run Python code in your browser without needing to install Python and all the libraries on your computer.

---

## 9. Step 4 — Install the Required Libraries

Run this in a Colab code cell:

```python
!pip install opencv-python-headless numpy matplotlib scikit-learn joblib
```

### Why install libraries?

Libraries contain reusable functions. OpenCV helps us process images, NumPy manages numerical data, and Scikit-learn provides the SVM classifier and evaluation tools.

Verify the installation:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import joblib

print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
print("Scikit-learn:", sklearn.__version__)
print("Libraries imported successfully.")
```

If no import errors appear, continue.

---

## 10. Step 5 — Upload and Extract the Dataset ZIP

Run this code in Google Colab:

```python
from google.colab import files

uploaded = files.upload()
```

Select the ZIP file you downloaded from GitHub.

Now extract it:

```python
import zipfile, os
with zipfile.ZipFile("dataset.zip", 'r') as zip_ref:
    zip_ref.extractall("dataset")

# Handle the extra wrapper folder from zipping
root = "dataset"
contents = os.listdir(root)
if len(contents) == 1 and os.path.isdir(os.path.join(root, contents[0])):
    root = os.path.join(root, contents[0])

classes = sorted(os.listdir(root))
print("Using folder:", root)
print("Classes found:", classes)   # should now print ['0','1','2','3','4','5']
```

### Check the folder structure

The dataset should contain six folders:

```text
dataset/
├── 0/
├── 1/
├── 2/
├── 3/
├── 4/
└── 5/
```

Each folder should contain images representing the corresponding class.

For example, folder `2` should contain images showing two fingers.

**Important:** The ZIP may contain an additional parent folder. Inspect the extracted directories and identify the folder that directly contains the six class folders.

---

## 11. Step 6 — Set the Dataset Path

Update the path according to the actual folder structure in Colab.

For example, if the six folders are directly inside `/content/dataset`:

```python
DATASET_DIR = "/content/dataset"

import os

print(os.listdir(DATASET_DIR))
```

The output should include:

```text
['0', '1', '2', '3', '4', '5']
```

The order of the folders in the output may differ.

If the folders are located elsewhere, change `DATASET_DIR` to the correct location.

---

## 12. Step 7 — Inspect the Dataset

Before training a model, verify that the expected images exist.

### Why is this important?

A missing folder, unreadable image, or incorrect label can cause errors or reduce the model's performance.

Run:

```python
import os

# Define the dataset directory using the 'root' variable from the previous cell
DATASET_DIR = root if 'root' in globals() else "dataset/fingers_final"

CLASSES = ["0", "1", "2", "3", "4", "5"]

for class_name in CLASSES:
    folder = os.path.join(DATASET_DIR, class_name)

    if not os.path.isdir(folder):
        print(f"Missing folder: {folder}")
        continue

    images = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

    print(f"Class {class_name}: {len(images)} images")
```

Confirm that all six folders exist and contain images.

### Dataset quality guidelines

- Use clear images with correctly labeled gestures.
- Include different hands, backgrounds, distances, and lighting conditions where possible.
- Keep the number of examples reasonably balanced across classes.
- Avoid duplicate or near-identical images across training and testing subsets.
- As a starting point, aim for approximately 30–50 images per class for this exercise, then expand the dataset.
- Use images you have permission to use.

A larger, more varied dataset can help, but image quantity alone does not guarantee accuracy.

---

## 13. Step 8 — Understand Images as Numerical Data

A computer represents an image as a grid of pixels.

For a typical color image, each pixel contains three color-channel values. OpenCV reads color images in BGR order by default.

### View a sample image

```python
import cv2
import matplotlib.pyplot as plt
import os

sample_folder = os.path.join(DATASET_DIR, "3")

sample_files = [
    f for f in os.listdir(sample_folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
]

if not sample_files:
    raise ValueError("No images found in class 3.")

sample_path = os.path.join(sample_folder, sample_files[0])
image = cv2.imread(sample_path)

if image is None:
    raise ValueError("Image could not be loaded.")

rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(rgb_image)
plt.title("Sample hand gesture")
plt.axis("off")
plt.show()

print("Image shape:", image.shape)
print("Example pixel:", image[0, 0])
```

### Why do this?

Viewing sample images helps you check whether the images match their folder labels and whether the data is suitable for training.

---

## 14. Step 9 — Resize Images

Images may have different dimensions. For this project, resize each image to 64 × 64 pixels.

### Why resize?

The model expects every example to have the same number of input features.

Run:

```python
resized = cv2.resize(image, (64, 64))

print("Original shape:", image.shape)
print("Resized shape:", resized.shape)

plt.imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
plt.title("Resized image: 64 x 64")
plt.axis("off")
plt.show()
```

A color image of size 64 × 64 contains:

64 × 64 × 3 = 12,288 pixel values.

Resizing makes the dimensions consistent, but it can remove image details.

---

## 15. Step 10 — Flatten Images into Features

### What is a feature?

A feature is a numerical input used by a machine learning model.

In this baseline project, the pixel values are the features.

The resized image has three dimensions: height, width, and color channels. Flattening converts the array into a single one-dimensional list of values.

Run:

```python
features = resized.flatten()

print("Number of features:", features.shape[0])
```

Expected output:

```text
Number of features: 12288
```

### Why flatten the image?

The SVM classifier expects each image to be represented as a feature vector. Flattening provides that representation.

The model does not explicitly understand fingers or hand anatomy. It learns patterns from the numerical values.

---

## 16. Step 11 — Normalize Pixel Values

Most standard 8-bit image pixels have values between 0 and 255.

We divide these values by 255 to scale them to the range 0–1.

### Why normalize?

Feature scaling can help machine learning algorithms work more effectively. SVM classifiers can be sensitive to the scale of their input features.

Run:

```python
import numpy as np

features = features.astype(np.float32) / 255.0

print("Minimum pixel value:", features.min())
print("Maximum pixel value:", features.max())
```

For a typical nonempty image, the normalized values will lie between 0 and 1.

**Important:** Use exactly the same preprocessing method for training images and future predictions.

---

## 17. Step 12 — Load the Complete Dataset

Now combine image loading, resizing, flattening, normalization, and labeling into one function.

### Why do this?

The model needs two arrays:

- **X:** Numerical features extracted from the images.
- **y:** Correct class labels for those images.

Run:

```python
import os
import cv2
import numpy as np

def load_dataset(dataset_dir):
    X = []
    y = []

    for class_name in ["0", "1", "2", "3", "4", "5"]:
        folder = os.path.join(dataset_dir, class_name)

        if not os.path.isdir(folder):
            raise FileNotFoundError(
                f"Missing class folder: {folder}"
            )

        for filename in os.listdir(folder):
            if not filename.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp")
            ):
                continue

            image_path = os.path.join(folder, filename)
            image = cv2.imread(image_path)

            if image is None:
                print("Skipping unreadable image:", image_path)
                continue

            image = cv2.resize(image, (64, 64))
            features = image.astype(np.float32).flatten() / 255.0

            X.append(features)
            y.append(int(class_name))

    if len(X) == 0:
        raise ValueError(
            "No images were loaded. Check your dataset path."
        )

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.int64)
    )

X, y = load_dataset(DATASET_DIR)

print("Feature matrix shape:", X.shape)
print("Label array shape:", y.shape)
print("Classes found:", np.unique(y))
```

### Understand the output

If 240 valid images are loaded, the feature matrix should have shape:

`(240, 12288)`

The label array should have shape:

`(240,)`

These are examples. Your actual values depend on your dataset.

**Checkpoint:** Confirm that all six classes appear in the output before training.

---

## 18. Step 13 — Split the Dataset into Training and Testing Data

We will use approximately 80% of the data for training and 20% for testing.

### Why split the data?

The model learns from the training data. The test data helps us estimate how well the model predicts examples that were not used to fit it.

Run:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
```

### Understand the parameters

- `test_size=0.20`: Reserves approximately 20% for testing.
- `random_state=42`: Makes the split reproducible.
- `stratify=y`: Attempts to preserve class proportions across the subsets.

**Important:** If the same person's nearly identical images appear in both sets, the evaluation may be overly optimistic. A stronger evaluation uses separate image collection sessions or people for testing when possible.

---

## 19. Step 14 — Understand the SVM Model

SVM stands for **Support Vector Machine**.

It is a supervised machine learning algorithm used for classification and other tasks.

Imagine that you have examples from two classes. An SVM tries to find a decision boundary that separates them. In this project, the classifier learns to distinguish among six gesture classes.

### Why use SVM?

- It is a useful introductory classification algorithm.
- It is available in Scikit-learn.
- It provides a baseline for experimenting with image classification.

### What are its limitations?

Flattened pixel features do not explicitly capture hand shape or spatial structure. The classifier may struggle when lighting, backgrounds, orientation, or hand position differ substantially from the training images.

More advanced models, such as convolutional neural networks or hand-landmark-based models, may perform better on varied real-world images.

---

## 20. Step 15 — Train the Model

Run:

```python
from sklearn.svm import SVC

model = SVC(kernel="linear")
model.fit(X_train, y_train)

print("Model training completed.")
```

### What happens here?

- `SVC` creates a Support Vector Classifier.
- `kernel="linear"` selects a linear decision function.
- `model.fit(X_train, y_train)` trains the model using the training features and their correct labels.

Training does not guarantee that the model has learned useful patterns. We must evaluate it on the separate test subset.

---

## 21. Step 16 — Make Predictions

Use the test features to predict gesture labels.

Run:

```python
y_pred = model.predict(X_test)

for actual, predicted in zip(y_test[:10], y_pred[:10]):
    print(f"Actual: {actual}, Predicted: {predicted}")
```

### How to interpret the output

- If actual and predicted values match, that prediction is correct.
- If they differ, the model has misclassified the image.

Do not judge the model using only a few examples. Evaluate all test predictions.

---

## 22. Step 17 — Calculate Test Accuracy

Accuracy is the proportion of test examples classified correctly.

Formula:

```text
Accuracy = Correct Predictions / Total Predictions
```

Run:

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print(f"Test accuracy: {accuracy:.2%}")
```

For example, 36 correct predictions out of 48 would give 75% accuracy.

Your result may be higher or lower. Record the actual result produced by your experiment.

### Why is accuracy not enough?

Overall accuracy may hide poor performance on a particular class. A confusion matrix helps us investigate which gestures the model confuses.

---

## 23. Step 18 — Generate a Confusion Matrix

Run:

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

labels = [0, 1, 2, 3, 4, 5]

cm = confusion_matrix(y_test, y_pred, labels=labels)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

display.plot(cmap="Blues", values_format="d")
plt.title("Hand Gesture Recognition — Confusion Matrix")
plt.show()
```

### How to read it

- Rows represent actual classes.
- Columns represent predicted classes.
- Diagonal cells represent correct predictions.
- Off-diagonal cells represent mistakes.

For example, if several actual class-2 images are predicted as class 3, investigate those images. They may be unclear, mislabeled, or visually similar under the current preprocessing.

---

## 24. Step 19 — Investigate Incorrect Predictions

Do not stop after calculating accuracy. Examine the mistakes.

Ask yourself:

1. Are some images incorrectly labeled?
2. Are particular classes underrepresented?
3. Are hands too small or partially outside the frame?
4. Are backgrounds or lighting conditions inconsistent?
5. Are training and testing images too similar?
6. Does the model confuse two gestures more frequently than others?

### Possible improvements

- Collect more varied images.
- Correct incorrect labels.
- Improve class balance.
- Make the hand easier to see in the image.
- Try a different model or image representation.
- Evaluate changes using a proper validation process.

If you repeatedly adjust the model based on test results, use a separate validation set for tuning and reserve a final test set for the final evaluation.

---

## 25. Step 20 — Predict a New Hand Image

Now test the model on a new image that was not part of the training or testing dataset.

Upload the image:

```python

from google.colab import files
print("Take a photo of your hand showing some fingers, then upload it")
uploaded = files.upload()
fname = list(uploaded.keys())[0]

test_img = cv2.imread(fname)
test_img = cv2.resize(test_img, (IMG_SIZE, IMG_SIZE))
test_input = test_img.flatten() / 255.0

prediction = model.predict([test_input])[0]
plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
plt.title(f"Model says: {prediction} fingers")
plt.axis('off')
plt.show()

```

### Important limitation

This baseline model predicts from the entire image. It does not automatically locate and crop the hand. A cluttered background or a hand positioned differently from the training examples may reduce accuracy.

---

## 26. Step 21 — Save the Trained Model

Run:

```python
import joblib

joblib.dump(model, "hand_gesture_svm.joblib")

print("Model saved successfully.")
```

To download the model:

```python
from google.colab import files

files.download("hand_gesture_svm.joblib")
```

### Why save the model?

Saving the fitted classifier lets you reuse it without retraining every time.

Preserve the preprocessing settings, class mapping, and required library versions as well. Future predictions must use the same image size, color-channel convention, feature order, and scaling as training.

**Security note:** Only load Joblib or pickle-based model files from trusted sources because loading such files can execute malicious code.

---

## 27. Step 22 — Create requirements.txt

Create a file named `requirements.txt` in your GitHub repository:

```text
opencv-python-headless
numpy
matplotlib
scikit-learn
joblib
```

### Why?

This records the main Python libraries needed to reproduce the project. For stronger reproducibility, record and test specific dependency versions after confirming the notebook works.

---

## 28. Step 23 — Create a .gitignore File

Create a file named `.gitignore`:

```text
__pycache__/
.ipynb_checkpoints/
.env
*.pyc
```

Do not upload passwords, API keys, private images, or other sensitive information.

Large datasets and model files may be unsuitable for ordinary Git commits. Follow the sharing method specified by your instructor if a file is too large.

---

## 29. Step 24 — Upload Your Work to GitHub

### Option A: Upload through the GitHub website

1. Open your repository.
2. Click **Add file → Upload files**.
3. Upload your completed notebook and supporting files.
4. Enter a meaningful commit message, such as `Add hand gesture recognition notebook`.
5. Click **Commit changes**.

### Option B: Use Git locally

Clone your repository:

```bash
git clone https://github.com/YOUR-USERNAME/hand-gesture-recognition.git
cd hand-gesture-recognition
```

Replace `YOUR-USERNAME` with your GitHub username.

After adding or updating your files:

```bash
git add README.md hand_gesture_recognition.ipynb requirements.txt .gitignore
git commit -m "Add hand gesture recognition project"
git push
```

If you work entirely in Google Colab, download your completed notebook and upload it using Option A or another approved GitHub workflow.

### Recommended repository structure

```text
hand-gesture-recognition/
├── README.md
├── hand_gesture_dataset.zip
├── hand_gesture_recognition.ipynb
├── requirements.txt
└── .gitignore
```

The ZIP filename is an example. Use the actual dataset filename.

**GitHub file-size note:** Browser uploads have a 25 MiB per-file limit. Files larger than 100 MiB are blocked in regular Git repositories. If your ZIP exceeds the supported limit, use Git LFS or an approved external dataset link instead.

---

## 30. Step 25 — Document Your Actual Results

After completing the experiment, record your results in the README or notebook.

Include:

- Number of images in each class.
- Preprocessing method and image dimensions.
- Training/testing split.
- Model and its settings.
- Actual test accuracy.
- Confusion matrix.
- At least two observations about incorrect predictions.
- Improvements you would make in a future version.

### Results template

| Item | Your result |
|---|---|
| Total valid images | Fill in after loading |
| Images per class | Fill in after inspection |
| Image size | 64 × 64 |
| Model | Linear SVM |
| Training proportion | 80% |
| Testing proportion | 20% |
| Test accuracy | Fill in after evaluation |
| Most frequently confused classes | Fill in after evaluation |
| Proposed improvement | Fill in after analysis |

Never present example values as measured results. Your report must reflect your own experiment.

---

## 31. Troubleshooting Guide

### Error: Image could not be loaded

**Possible cause:** The path is incorrect, the filename differs, or the image format is unsupported.

**Solution:** Print the directory contents, check the path, and try a valid JPG or PNG image.

### Error: Missing class folder

**Possible cause:** The ZIP extracted into a nested folder.

**Solution:** Inspect the extracted directories and update `DATASET_DIR`.

### Error: No images were loaded

**Possible cause:** The dataset path is incorrect, or the images use unsupported formats.

**Solution:** Check the path, inspect the folder contents, and verify the supported file extensions.

### Error: Only one class is found

**Possible cause:** Images are in the wrong folders or the dataset did not extract correctly.

**Solution:** Check all six class folders and verify their image counts.

### Accuracy is low

**Possible cause:** Too few images, incorrect labels, class imbalance, inconsistent framing, or limitations of the baseline model.

**Solution:** Inspect misclassified images, improve data quality, and evaluate changes carefully.

### New images are predicted incorrectly

**Possible cause:** The new image differs from training data in background, lighting, hand position, scale, or orientation.

**Solution:** Use consistent framing for the initial demonstration, then expand the training data with more varied examples.

---

## 32. Student Submission Checklist

Before submitting, verify that you have completed the following:

- [ ] Created a GitHub repository.
- [ ] Added the README.md file.
- [ ] Opened the provided Google Colab notebook.
- [ ] Saved your own notebook copy.
- [ ] Downloaded and extracted the dataset ZIP.
- [ ] Verified all six class folders.
- [ ] Inspected sample images.
- [ ] Resized, flattened, and normalized the images.
- [ ] Created the feature matrix and label array.
- [ ] Split the data into training and testing subsets.
- [ ] Trained the SVM classifier.
- [ ] Generated predictions on the test set.
- [ ] Calculated actual test accuracy.
- [ ] Generated and interpreted a confusion matrix.
- [ ] Tested a separate new image.
- [ ] Saved the trained model.
- [ ] Documented the results and limitations.
- [ ] Checked that no private data or credentials were committed.

---

## 33. Final Deliverables

Submit the following to your instructor:

1. **GitHub repository URL** — containing the README and supporting files.
2. **Completed Colab/Jupyter notebook** — showing code, outputs, predictions, and evaluation.
3. **Evaluation evidence** — actual accuracy and confusion matrix.
4. **Model file** — only if requested, shared through the approved method.
5. **Short reflection** — what worked, what failed, and what you would improve.

---

## 34. Reflection Questions

Answer these questions in your notebook or final report:

1. What is the difference between traditional programming and machine learning?
2. What is supervised learning?
3. Why do we need labeled images?
4. Why must images have consistent dimensions?
5. What does flattening an image mean?
6. Why do we divide pixel values by 255?
7. What is the purpose of the training/testing split?
8. What does an SVM do?
9. What does test accuracy tell us?
10. How does a confusion matrix help us?
11. Why might a model perform well on test images but poorly on a live photograph?
12. What changes might improve the model?
13. Why must preprocessing remain consistent during training and prediction?
14. Why should we not use the test set repeatedly to tune the model?

---

## 35. Conclusion

You have explored the fundamental steps of a supervised machine learning project: image loading, preprocessing, feature extraction, model training, prediction, evaluation, and model export.

Remember:

**A working model is not necessarily a reliable model.**

Measure its performance honestly, investigate its errors, document your results, and explain its limitations.

Your goal is not just to make a prediction. Your goal is to understand how the prediction was produced and how the model could be improved.

**Learn the process. Understand the code. Evaluate the results. Document your work.**
