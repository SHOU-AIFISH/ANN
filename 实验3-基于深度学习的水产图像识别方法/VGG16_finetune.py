import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt
from keras.applications.resnet50 import preprocess_input
from keras import applications
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D,Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau

TRAIN_DIR = r'G:\tree1\train'  # 训练集数据
VAL_DIR = r'G:\tree1\val'  # 验证集数据
OUT_PUT_MODEL_FT = 'vggtree_15.h5'
IM_WIDTH, IM_HEIGHT = 224, 224
NB_EPOCHS = 100
BATCH_SIZE = 32
FC_SIZE = 512
NB_IV3_LAYERS_TO_FREEZE = 15


def get_nb_files(directory):
    '''获取数据集信息'''
    if not os.path.exists(directory):
        return 0
    count = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            count += len(glob.glob(os.path.join(r, dr + "/*")))
    return count


def setup_to_tansfer_learn(model, base_model):
    '''冻结全部CONV层并编译'''
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


def add_new_last_layer(base_model, nb_classes):
    '''添加fc层'''
    x = base_model.output
    # x = GlobalAveragePooling2D()(x)
    x = Flatten(name='flatten')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(nb_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def setup_to_finetune(model):
    '''冻结除最后block之外的层，并编译'''
    for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
        layer.trainable = False
    for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
        layer.trainable = True
    model.compile(optimizer=SGD(lr=1e-4, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])


def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'b.', label='Train_acc')
    plt.plot(epochs, val_acc, 'r-', label='Val_acc')
    plt.title('Training and validation accuracy')
    plt.legend(loc='lower right')
    plt.figure()
    plt.plot(epochs, loss, 'b.', label='Train_loss')
    plt.plot(epochs, val_loss, 'r-', label='Val_loss')
    plt.title('Training and validation loss')
    plt.legend(loc='upper right')
    plt.show()


def train(args):
    '''先模式一训练，后finetune训练'''
    nb_train_samples = get_nb_files(args.train_dir)
    nb_classes = len(glob.glob(args.train_dir + "/*"))
    nb_val_samples = get_nb_files(args.val_dir)
    nb_epoch = int(args.nb_epoch)
    batch_size = int(args.batch_size)

    # 生成数据集
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    train_generator = train_datagen.flow_from_directory(
        args.train_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size,
    )
    validation_generator = val_datagen.flow_from_directory(
        args.val_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size,
    )

    # 配置model
    base_model = applications.VGG16(weights='imagenet', input_shape=(224, 224, 3), include_top=False)
    model = add_new_last_layer(base_model, nb_classes)
    best_model_ft = ModelCheckpoint(OUT_PUT_MODEL_FT, monitor='val_acc', verbose=1, save_best_only=True)
    learningRate = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, mode='auto',
                                      epsilon=0.0001, cooldown=0, min_lr=0)
    # finetune
    setup_to_finetune(model)
    model.summary()
    print('start FT')
    history_ft = model.fit_generator(
        train_generator,
        # TRAIN_DIR,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=nb_epoch,
        # validation_data=VAL_DIR,
        validation_data=validation_generator,
        validation_steps=nb_val_samples // batch_size,
        class_weight='auto',
        verbose=1,
        callbacks=[best_model_ft,learningRate]
    )
    print('FT done')
    model.save(args.output_model_file)

    model.summary()
    plot_training(history_ft)

if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--train_dir", default=TRAIN_DIR)
    a.add_argument("--val_dir", default=VAL_DIR)
    a.add_argument("--nb_epoch", default=NB_EPOCHS)
    a.add_argument("--batch_size", default=BATCH_SIZE)
    a.add_argument("--output_model_file", default=OUT_PUT_MODEL_FT)
    a.add_argument("--plot", action="store_true")
    args = a.parse_args()
    if args.train_dir is None or args.val_dir is None:
        a.print_help()
        sys.exit(1)

    if (not os.path.exists(args.train_dir)) or (not os.path.exists(args.val_dir)):
        print("Directory not found")
        sys.exit(1)

    train(args)

