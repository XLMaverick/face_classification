
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

from models.cnn import mini_XCEPTION
from models.cnn import big_XCEPTION
from utils.datasets import DataManager
from utils.datasets import _load_train_ferplus
from utils.datasets import _load_valid_ferplus
from utils.preprocessor import preprocess_input
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


# parameters
batch_size = 32
num_epochs = 1000
input_shape = (64, 64, 1)
num_classes = 8
patience = 50
base_path = '../trained_models/emotion_models/perplus_trained_model/'

# data generator
data_generator = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True)

# model parameters/compilation
model = mini_XCEPTION(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()


datasets = ['ferplus']
for dataset_name in datasets:
    print('Training dataset:', dataset_name)

    # callbacks
    log_file_path = base_path + dataset_name + '_emotion_training.log'
    csv_logger = CSVLogger(log_file_path, append=False)
    early_stop = EarlyStopping('val_loss', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                                  patience=int(patience/4), verbose=1)
    trained_models_path = base_path + dataset_name + 'mini_XCEPTION'
    model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1,
                                                    save_best_only=True)
    callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

    # loading dataset
    data_loader = DataManager(dataset_name, image_size=input_shape[:2])
    data_path = data_loader.dataset_path
    image_size = data_loader.image_size
    
    #train_faces,train_emotions,valid_faces,valid_emotions = data_loader.get_data()
    train_faces,train_emotions = _load_train_ferplus(data_path,image_size)
    valid_faces,valid_emotions = _load_valid_ferplus(data_path,image_size)
    train_faces = preprocess_input(train_faces)
    valid_faces = preprocess_input(valid_faces)
    num_samples, num_classes = train_emotions.shape
    train_data = train_faces,train_emotions 
    val_data = valid_faces,valid_emotions
    model.fit_generator(data_generator.flow(train_faces, train_emotions,
                                            batch_size),
                        steps_per_epoch=len(train_faces) / batch_size,
                        epochs=num_epochs, verbose=1, callbacks=callbacks,
                        validation_data=val_data)
