import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_and_preprocess_data(image_size=(128, 128), batch_size=32):
    """
    Loads and preprocesses image data from the 'data/images' directory.

    Args:
        image_size (tuple): The target size for resizing images (width, height).
        batch_size (int): The number of images per batch.

    Returns:
        tuple: A tuple containing train, validation, and test data generators.
    """
    # Data augmentation and rescaling for training data
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    # Rescaling for validation and test data (no augmentation)
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        'data/images/train',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    validation_generator = val_test_datagen.flow_from_directory(
        'data/images/val',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_generator = val_test_datagen.flow_from_directory(
        'data/images/test',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False # Keep data in order for evaluation
    )

    return train_generator, validation_generator, test_generator

if __name__ == '__main__':
    train_gen, val_gen, test_gen = load_and_preprocess_data()
    print("Data generators created successfully.")
    print(f"Training classes: {train_gen.class_indices}")
    print(f"Validation classes: {val_gen.class_indices}")
    print(f"Test classes: {test_gen.class_indices}")
