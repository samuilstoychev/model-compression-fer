from tensorflow.keras.preprocessing.image import ImageDataGenerator
DATASET_DIR = "/Users/samuilstoychev/ckplus_cross_subject_cropped/"
EMOTION_MAPPING = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]

def load_ckplus(mode):
    """Load the CK+ dataset. `mode` can be 'train' or 'test' denoting the data split."""
    img_size = 48
    batch_size = 32

    datagen = ImageDataGenerator(horizontal_flip=True if mode=="train" else False, rotation_range=10 if mode=="train" else False)
    generator = datagen.flow_from_directory(DATASET_DIR + mode + "/",
                                            target_size=(img_size, img_size),
                                            color_mode="grayscale",
                                            batch_size=batch_size,
                                            class_mode='categorical',
                                            shuffle=True if mode=="train" else False)
    return generator