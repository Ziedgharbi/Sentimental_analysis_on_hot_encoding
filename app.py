import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as ft
import tensorflow.keras as keras
import tensorflow.keras.datasets.imdb as imdb

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report

project_dir= "C:/Users/pc/Nextcloud/Python/GITHUB/Sentimental_analysis_on_hot_encoding/"
#data_dir=project_dir+"data/"

vocab_size=20000

hide_most_frequently=0

epochs=10
batch_size=512

"exemple one hot encoding"
sentence= "  I really liked the movie is so nice   "
sentence =  sentence.lower().split() 

# create dictionnary ---> word : number
dictionnary={"a":0, "nice":1, "great":2, "really":3, "movie":4, "liked":5, "fantastic":6, "is":7,
             "i":8, "the":9, "so":10, }

# encoding dictonnary into numercial vector
sentence_vector=[dictionnary[w] for w in sentence ]

print(" sentence in words is : ", sentence)
print(" sentence in numerical : ", sentence_vector)

# one hot encod now 

onehote_mat=np.zeros((len(dictionnary),len(sentence)))

for i,w in enumerate(sentence_vector):
    onehote_mat[w,i]=1
    
print(onehote_mat, "\n")
data= {  f'{sentence[i]:.^10}':onehote_mat[:,i] for i,w in enumerate(sentence_vector)  }
df=pd.DataFrame(data)
df.index=dictionnary.keys()
df.style.format(precision=0).highlight_max(axis=0).set_properties(**{'text-align' :'center'})

"""--------------- real exemple --------------"""

#load data 
(x_train, y_train), (x_test, y_test) =  imdb.load_data(num_words=vocab_size, skip_top=hide_most_frequently)

y_train=np.asanyarray(y_train).astype('float32')
y_test=np.asanyarray(y_test).astype('float32')

x_train.shape, x_test.shape, y_train.shape, y_test.shape

x_train[15]
y_train

# load dictionnary
dictionary= imdb.get_word_index()

# add 4 words ; "<pad>":0, "<start>":1, "<unknown>":2, "<undef>":3
dictionary= {w:i+3 for w,i in dictionary.items()}

dictionary.update({"<pad>":0, "<start>":1, "<unknown>":2, "<undef>":3})

#reverse dictionary
dictionary_reverse= {i:w for w,i in dictionary.items() }


#see some exemple 
for i in range(350,360):
    print(i," : ",dictionary_reverse[i] )
    
print( " exemple as vector : \n",x_train[12] , "\n")

indice=x_train[12]
print("\n exemple as humain :\n", ' '.join([dictionary_reverse[i] for i in indice]))

# see the distribution of review size

sizes=[len(i) for i in x_train ]

plt.hist(sizes, bins=400)
plt.title( "Distibution of size [min, max] : [" + str(min(sizes))+" , "+str(max(sizes))+" ]" )
plt.show()

# Vectorizing x_train:
    
one_hote_mat=np.zeros((x_train.shape[0], vocab_size))

for i, sentence in enumerate(x_train):
    for word in sentence :
        one_hote_mat[i,word]=1
x_train=one_hote_mat        

# Vectorizing x_test:
    
one_hote_mat=np.zeros((x_test.shape[0], vocab_size))

for i, sentence in enumerate(x_test):
    for word in sentence :
        one_hote_mat[i,word]=1
x_test=one_hote_mat        

# model devolepment 

model=keras.Sequential()

model.add(keras.layers.Input(shape=(vocab_size,)))
model.add(keras.layers.Dense(32,activation='relu'))
model.add(keras.layers.Dense(32,activation='relu'))
model.add(keras.layers.Dense(1,activation='sigmoid'))

model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])

model.summary()

## create callback to save best model
save_model=ft.keras.callbacks.ModelCheckpoint(filepath=project_dir+"model/best_model.h5", save_best_only=True)

## train model 

history=model.fit(x_train, y_train, batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test,y_test),
          callbacks=[save_model])

history.history['val_accuracy']
history.epoch

plt.subplot(2,1,1)
plt.plot(history.epoch, history.history['loss'], label="Loss train")
plt.plot(history.epoch, history.history['val_loss'], label="Loss test")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(2,1,2)
plt.plot(history.epoch, history.history['accuracy'], label="Accuracy train")
plt.plot(history.epoch, history.history['val_accuracy'], label="Accuracy test")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.show()

# load model 

model=keras.models.load_model(project_dir+"model/best_model.h5")

model.evaluate(x_test, y_test)

y_pred=model.predict(x_test)

## confusion matrix plo

y_pred= [ 1 if x>=0.5 else 0 for x in y_pred]

confu_matrix= confusion_matrix(y_test, y_pred)

disp=ConfusionMatrixDisplay(confu_matrix)
disp.plot()

report=classification_report(y_test, y_pred)
print(report)
