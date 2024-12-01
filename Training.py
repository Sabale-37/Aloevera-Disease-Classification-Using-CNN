# Importing Libraries
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import label_binarize
from itertools import cycle
from tensorflow.keras.preprocessing.image import ImageDataGenerator

## Start the timer to calculate elapsed time
start_time = time.time()
# Importing Dataset
# Define the path to the dataset
data_dir = 'D:/Projects/Medicinal-Plant-Diseases-Detection/Dataset/Augmented Aloe Vera Dataset'

# Image parameters
img_height, img_width = 128, 128
batch_size = 32
num_classes = 3  # Healthy, Rot, Rust

# Data Preprocessing
# Initialize ImageDataGenerator for training with validation split
train_val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # 20% of data for validation
)

# Define train and validation generators
train_generator = train_val_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'  # Training data
)

val_generator = train_val_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'  # Validation data
)

# Building the CNN model
model = Sequential([
    # Convolutional Block 1
    Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.001), input_shape=(img_height, img_width, 3)),
    MaxPooling2D((2, 2)),
    Dropout(0.1),  

    # Convolutional Block 2
    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    MaxPooling2D((2, 2)),
    Dropout(0.2),

    # Convolutional Block 3
    Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    # Convolutional Block 4
    Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    MaxPooling2D((2, 2)),
    Dropout(0.4),

    # Fully Connected Layers
    Flatten(),
    Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  # Output layer
])

# Compile the model with a fixed learning rate for stability
model.compile(
    optimizer=Adam(learning_rate=0.0005),  # Slightly higher learning rate for faster convergence
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

### Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
# Fit the model on the data
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size,
    epochs=15,  
    callbacks=[early_stopping]
)
### Evaluate the model on the validation set
val_loss, val_acc = model.evaluate(val_generator, steps=val_generator.samples // batch_size)
print(f"Validation Accuracy: {val_acc:.2f}")

### Save the Model
model.save('aloe_vera_model.h5')
# classification report
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=["Healthy", "Rot", "Rust"]))
# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Healthy", "Rot", "Rust"], yticklabels=["Healthy", "Rot", "Rust"])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')

# Plotting ROC curve
# Binarize the labels for multiclass ROC
y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
y_pred_bin = label_binarize(y_pred, classes=[0, 1, 2])

# Calculate ROC curve and AUC for each class
fpr, tpr, roc_auc = {}, {}, {}
class_names = ["Healthy", "Rot", "Rust"]

for i in range(3):  # 3 classes: Healthy (0), Rot (1), Rust (2)
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve for all classes
plt.figure(figsize=(10, 8))
for i in range(3):
    plt.plot(fpr[i], tpr[i], label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')

# Plot diagonal line (random classifier)
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')

# Labels and Title
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)

### Calculating elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed Time: {elapsed_time:.2f} seconds")
