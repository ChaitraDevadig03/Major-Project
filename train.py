import os
from data_preprocessing import load_and_preprocess_data
from model import build_cnn_model
import tensorflow as tf

def train_model(epochs=10, image_size=(128, 128), batch_size=32):
    """
    Trains the CNN model using the preprocessed image data.

    Args:
        epochs (int): The number of epochs to train the model.
        image_size (tuple): The target size for resizing images (width, height).
        batch_size (int): The number of images per batch.
    """
    train_generator, validation_generator, test_generator = load_and_preprocess_data(
        image_size=image_size, batch_size=batch_size
    )

    num_classes = len(train_generator.class_indices)
    print(f"Number of classes detected: {num_classes}")

    model = build_cnn_model(input_shape=(image_size[0], image_size[1], 3), num_classes=num_classes)
    model.summary()

    print("Starting model training...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size
    )
    print("Model training finished.")

    # Save the trained model
    model_save_path = 'crop_prediction_model.keras'
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    # Evaluate the model on the test set
    print("Evaluating model on test data...")
    test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    return history, model

if __name__ == '__main__':
    # Ensure TensorFlow uses GPU if available
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU is available and being used.")
    else:
        print("GPU not available, using CPU.")

    # You can adjust epochs, image_size, and batch_size here
    history, trained_model = train_model(epochs=10, image_size=(128, 128), batch_size=32)
