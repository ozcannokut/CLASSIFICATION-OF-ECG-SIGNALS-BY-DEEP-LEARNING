import sys
from PyQt5.QtWidgets import QApplication, QMainWindow,QFileDialog
from form_python import Ui_Widget
from PyQt5.QtCore import QObject, QThread, pyqtSignal,Qt
import pandas as pd
from PyQt5.QtGui import QPixmap
################################3


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical
from sklearn.utils import class_weight
import warnings
from sklearn.utils import resample
warnings.filterwarnings('ignore')

from keras.layers import Dense, Convolution1D, MaxPool1D, Flatten, Dropout
from keras.layers import Input
from keras.models import Model
from tensorflow.keras.layers import BatchNormalization
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint



global fname_modelsec,fname_trainveriseti,fname_testveriseti
fname_modelsec=""
fname_trainveriseti=""
fname_testveriseti=""

class MainWindow(QMainWindow):
        
        def __init__(self): #classtan nesne türetme clasın ilk methodu olmak zorunda
        #class içinden türetmiş olduğumuz nesnelere ulaşmamızı sağlar.
            super(MainWindow, self).__init__()
            #self.setWindowFlags(Qt.WindowStaysOnTopHint)
            self.ui = Ui_Widget()
            self.ui.setupUi(self)
            self.ui.modelsec.clicked.connect(self.modelsec)
            self.ui.trainveriseti.clicked.connect(self.trainveriseti)
            self.ui.testveriseti.clicked.connect(self.testveriseti)
            self.ui.hesapla.clicked.connect(self.hesapla)
            ##################################################################
            
            
            
        def modelsec(self):
            global fname_modelsec,fname_trainveriseti,fname_testveriseti
            try:
                    
                fname_modelsec=QFileDialog.getOpenFileName(self, 'Open file', 'D:\codefirst.io\PyQt5 tutorials\Browse Files', 'Images (*.h5)')
                print(fname_modelsec[0])
                self.ui.model_label.setText(fname_modelsec[0])        
                fname_modelsec=str(fname_modelsec[0])
            except:
                pass
            
                
        def trainveriseti(self):
            global fname_modelsec,fname_trainveriseti,fname_testveriseti
            fname_trainveriseti=QFileDialog.getOpenFileName(self, 'Open file', 'D:\codefirst.io\PyQt5 tutorials\Browse Files', 'Images (*.csv)')
            print(fname_trainveriseti[0])
            self.ui.train_label.setText(fname_trainveriseti[0])
            fname_trainveriseti=str(fname_trainveriseti[0])
            
        def testveriseti(self):
            global fname_modelsec,fname_trainveriseti,fname_testveriseti
            fname_testveriseti=QFileDialog.getOpenFileName(self, 'Open file', 'D:\codefirst.io\PyQt5 tutorials\Browse Files', 'Images (*.csv)')
            print(fname_testveriseti[0])
            self.ui.test_label.setText(fname_testveriseti[0])
            fname_testveriseti=str(fname_testveriseti[0])
            
        def hesapla(self):
            global fname_modelsec,fname_trainveriseti,fname_testveriseti
            print("geldi")
            
            
            # Veri setlerinin alınması
            train_df=pd.read_csv(fname_trainveriseti,header=None)
            test_df=pd.read_csv(fname_testveriseti,header=None)

                        
            # Veri kümesi dengesi
            train_df[187]=train_df[187].astype(int)
            equilibre=train_df[187].value_counts()
            print(equilibre)
            
            
            # Bu görüntüde sınıfların dengesindeki büyük farkın altını çizebiliriz.
            plt.figure(figsize=(20,10))
            my_circle=plt.Circle( (0,0), 0.7, color='white')
            plt.pie(equilibre, labels=['n','q','v','s','f'], colors=['red','green','blue','skyblue','orange'],autopct='%1.1f%%')
            p=plt.gcf()
            p.gca().add_artist(my_circle)
            plt.show()
            
            
            
            df_1=train_df[train_df[187]==1]
            df_2=train_df[train_df[187]==2]
            df_3=train_df[train_df[187]==3]
            df_4=train_df[train_df[187]==4]
            df_0=(train_df[train_df[187]==0]).sample(n=20000,random_state=42)
            
            df_1_upsample=resample(df_1,replace=True,n_samples=20000,random_state=123)
            df_2_upsample=resample(df_2,replace=True,n_samples=20000,random_state=124)
            df_3_upsample=resample(df_3,replace=True,n_samples=20000,random_state=125)
            df_4_upsample=resample(df_4,replace=True,n_samples=20000,random_state=126)
            
            train_df=pd.concat([df_0,df_1_upsample,df_2_upsample,df_3_upsample,df_4_upsample])

            



            equilibre=train_df[187].value_counts()
            print(equilibre)
     
        
            
        
            c=train_df.groupby(187,group_keys=False).apply(lambda train_df : train_df.sample(1))
        
      
            
            def plot_hist(class_number,size,min_,bins):
                img=train_df.loc[train_df[187]==class_number].values
                img=img[:,min_:size]
                img_flatten=img.flatten()
            
                final1=np.arange(min_,size)
                for i in range (img.shape[0]-1):
                    tempo1=np.arange(min_,size)
                    final1=np.concatenate((final1, tempo1), axis=None)
                print(len(final1))
                print(len(img_flatten))
                plt.hist2d(final1,img_flatten, bins=(bins,bins),cmap=plt.cm.jet)
                plt.show()

            # Ön işleme
            
            # Bu bölümde verileri modele sokmadan önce nasıl işleyeceğimiz kısmı
            
            # Modelin başarı diğer veri setlerinde de başarılı olması için veri setindeki sinyallere
            # biraz gürültü ekledik
            def add_gaussian_noise(signal):
                noise=np.random.normal(0,0.5,186)
                return (signal+noise)
          
            
            # gürültülü sinyal ile normal sinyalin görselleştirilmesi
            tempo=c.iloc[0,:186]
            bruiter=add_gaussian_noise(tempo)

            
            
            
            target_train=train_df[187]
            target_test=test_df[187]
            y_train=to_categorical(target_train)
            y_test=to_categorical(target_test)
           
            
            X_train=train_df.iloc[:,:186].values
            X_test=test_df.iloc[:,:186].values
            for i in range(len(X_train)):
                X_train[i,:186]= add_gaussian_noise(X_train[i,:186])
            X_train = X_train.reshape(len(X_train), X_train.shape[1],1)
            X_test = X_test.reshape(len(X_test), X_test.shape[1],1)
            
           
          
            
            def evaluate_model(history,X_test,y_test,model):
                scores = model.evaluate((X_test),y_test, verbose=0)
                print("Accuracy: %.2f%%" % (scores[1]*100))
                
                print(history)
                fig1, ax_acc = plt.subplots()
                plt.plot(history['accuracy'])
                plt.plot(history['val_accuracy'])
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.title('Model - Accuracy')
                plt.legend(['Training', 'Validation'], loc='lower right')
                plt.savefig("gorsel.png")
                plt.show()
                
                pixmap = QPixmap("sonuc.png")
                pixmap=pixmap.scaled(681, 712)
                self.ui.gorsel_labeli.setPixmap(pixmap)              
                fig2, ax_loss = plt.subplots()
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Model- Loss')
                plt.legend(['Training', 'Validation'], loc='upper right')
                plt.plot(history['loss'])
                plt.plot(history['val_loss'])
                plt.show()
                target_names=['0','1','2','3','4']
                
                y_true=[]
                for element in y_test:
                    y_true.append(np.argmax(element))
                prediction_proba=model.predict(X_test)
                prediction=np.argmax(prediction_proba,axis=1)
                cnf_matrix = confusion_matrix(y_true, prediction)
                
                self.ui.dogrulukorani.setText(str("Accuracy: %.2f%%" % (scores[1]*100)))
           
            
            
            
            
            im_shape=(X_train.shape[1],1)
            inputs_cnn=Input(shape=(im_shape), name='inputs_cnn')
            conv1_1=Convolution1D(64, (6), activation='relu', input_shape=im_shape)(inputs_cnn)
            # çıktının her bir elemanına uygulanıcak fonksiyon relu parametre 
            #vektördeki özellikleri algılamak için kullanabilecek katmanı ifade eder.
            conv1_1=BatchNormalization()(conv1_1)
            #daha düzenli hale getirir.
            pool1=MaxPool1D(pool_size=(3), strides=(2), padding="same")(conv1_1)
            conv2_1=Convolution1D(64, (3), activation='relu', input_shape=im_shape)(pool1)
            conv2_1=BatchNormalization()(conv2_1)
            pool2=MaxPool1D(pool_size=(2), strides=(2), padding="same")(conv2_1)
            conv3_1=Convolution1D(64, (3), activation='relu', input_shape=im_shape)(pool2)
            conv3_1=BatchNormalization()(conv3_1)
            pool3=MaxPool1D(pool_size=(2), strides=(2), padding="same")(conv3_1)
            flatten=Flatten()(pool3)
            dense_end1 = Dense(64, activation='relu')(flatten)
            dense_end2 = Dense(32, activation='relu')(dense_end1)
            main_output = Dense(5, activation='softmax', name='main_output')(dense_end2)
             #çıktının boyutunu tanımlayan en yoğun katman   
                
            model = Model(inputs= inputs_cnn, outputs=main_output)
            model.compile(optimizer='adam', loss='categorical_crossentropy',metrics = ['accuracy'])
             #modeli derlemek için compile yöntemi sağlar
             
             #tahmin ve kayıp fonksiyonunu karşılaştırarak girdi ağırlıklarını optimize eden fonksiyondur.
             #öğrenme sürecindeki hatayı veya sapmayı bulmak için kullanılır.
    
    
            callbacks = [EarlyStopping(monitor='val_loss', patience=8),
             ModelCheckpoint(filepath=fname_modelsec, monitor='val_loss', save_best_only=True)]
            history=np.load('my_history.npy',allow_pickle='TRUE').item()    
            model.load_weights(fname_modelsec)
            y_pred=model.predict(X_test)
            
            
           
            # Modelin doğruluğunu görselleştirdik
            evaluate_model(history,X_test,y_test,model)
            y_pred=model.predict(X_test)      
        
     
        
if __name__ == "__main__":
        app = QApplication(sys.argv)

        window = MainWindow()
        window.setWindowTitle("EKG SİNYALLERİ ARAYÜZ")
        window.show()
    
        sys.exit(app.exec_()) #uygulama başlatma


