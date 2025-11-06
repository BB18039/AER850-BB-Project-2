#AER8520 Project 2
#Bryant Berrio 501162030
#Model Design File 


#1 Data Processing

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers, optimizers, callbacks

basedir=r"C:\Users\bryan\Documents\GitHub\AER850-BB-Project-2\Project 2 Data"
datadir=os.path.join(basedir, "Data") #file that has train, valid, test folders

traindir=os.path.join(datadir, "train") #train folder
valdir=os.path.join(datadir, "valid") #valid folder
testdir=os.path.join(datadir, "test") #test folder

imgsize=(500,500)
batchsize=32
seed=42

#Train --> rescale +light augmentation and Val --> rescale only=ImageDataGenerator(rescale only)

train_datagen=ImageDataGenerator(rescale=1./255,shear_range =0.15, zoom_range=0.15,horizontal_flip=True)

val_datagen=ImageDataGenerator(rescale=1./255)

train_gen=train_datagen.flow_from_directory(directory=traindir, target_size=imgsize,batch_size=batchsize, class_mode="categorical", shuffle=True, seed=seed)

val_gen =val_datagen.flow_from_directory(directory=valdir, target_size=imgsize, batch_size=batchsize, class_mode="categorical", shuffle=False)

print("Class indices:", train_gen.class_indices)
print("Train samples:", train_gen.n, "| Val samples:", val_gen.n)

#2 NN Design

# #Design for Model A

# def build_model_a(input_shape=(500,500,3),num_classes=3):
#     model=models.Sequential([
#         layers.Conv2D(32,(3,3),activation='relu',input_shape=input_shape), layers.Conv2D(64, (3,3), activation='relu'),
#         layers.MaxPooling2D((2,2)),
#         layers.Flatten(),
#         layers.Dense(128, activation='relu'),
#         layers.Dropout(0.3),
#         layers.Dense(num_classes, activation='softmax')
#         ])
   
#     model.compile(
#         optimizer=optimizers.Adam(learning_rate=1e-4),
#         loss='categorical_crossentropy',
#         metrics=['accuracy']
#         )
#     return model

# #Design for Model B
# def build_model_b(input_shape=(500,500,3),num_classes=3):
#     model=models.Sequential([
#         layers.Conv2D(32,(3,3),activation='relu', input_shape=input_shape),
#         layers.Conv2D(32,(3,3),activation='relu'),
#         layers.MaxPooling2D((2,2)),
#         layers.Dropout(0.25),
        
#         layers.Conv2D(64,(3,3),activation='relu'),
#         layers.Conv2D(64,(3,3), activation='relu'),
#         layers.MaxPooling2D((2,2)),
#         layers.Dropout(0.25),
        
#         layers.Flatten(),
#         layers.Dense(256, activation='relu'),
#         layers.Dropout(0.5),
#         layers.Dense(num_classes, activation='softmax')
#         ])
    
#     model.compile(optimizer=optimizers.Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

#     return model

# #building both models
# model_a= build_model_a(num_classes=train_gen.num_classes)
# model_b= build_model_b(num_classes=train_gen.num_classes)

# model_a.summary()
# model_b.summary()

#3 Hyperparameter Analysis
#import json

# #Training settings
# EPOCHS= 30
# steps_per_epoch= train_gen.n//train_gen.batch_size
# valsteps=val_gen.n//val_gen.batch_size
   

# cba=[callbacks.EarlyStopping(monitor="val_accuracy",patience=5,restore_best_weights=True),
#      callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
#      callbacks.ModelCheckpoint("best_model_a.h5", monitor="val_accuracy",save_best_only=True, verbose=1)]

# cbb=[callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
#     callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
#     callbacks.ModelCheckpoint("best_model_b.h5", monitor="val_accuracy", save_best_only=True, verbose=1)]

# print("\nTraining Model A...")
# hist_a=model_a.fit(train_gen, epochs=EPOCHS, steps_per_epoch=steps_per_epoch, validation_data=val_gen, validation_steps=valsteps, callbacks=cba, verbose=1)

# print("\nTraining Model B")
# hist_b=model_b.fit(train_gen, epochs=EPOCHS, steps_per_epoch=steps_per_epoch, validation_data=val_gen, validation_steps=valsteps, callbacks=cbb, verbose=1)

# #saving class index mapping for step 5
# with open ("class_indices.json", "w") as f:
#     json.dump(train_gen.class_indices, f)

# #outputting comparison metric
# bestvalacc_a=max(hist_a.history["val_accuracy"])
# bestvalacc_b=max(hist_a.history["val_accuracy"])

# print(f"\nBest Val Acc --> Model A: {bestvalacc_a:.4f} | Model B: {bestvalacc_b:.4f}")
# print ("Saved: best_model_a.h5, best_model_b.h5,class_indices.json")

#Step 4 Model Evaluation
def plot_history(history, title_prefix):
   
    #Accuracy plot
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val accuracy'],label='Val')
    plt.xlabel ('Epoch');plt.ylabel('Accuracy');plt.title(f'{title_prefix} Accuracy')
    plt.legend();plt.grid(True);plt.tight_layout()
    plt.savefig(f"{title_prefix.lower().replace(' ','_')}_loss.png", dpi=160)
    plt.show()

    #Loss plot
    plt.figure()
    plt.plot(history.history['loss'],label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.xlabel('Epoch');plt.grid(True);plt.title(f'{title_prefix} Loss')
    plt.legend;plt.grid(True);plt.tight_layout()
    plt.savefig(f"{title_prefix.lower().replace(' ','_')}_loss.png",dpi=160)
    plt.show()
    
    
    plot_history(hist_a,"Model A")
    plot_history(hist_b, "Model B")
    