# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'D:/masaüstü/yazılımileilgilihersey/onluk/ekg sinyalleri/form.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Widget(object):
    def setupUi(self, Widget):
        Widget.setObjectName("Widget")
        Widget.resize(1200, 659)
        Widget.setStyleSheet("")
        self.train_label = QtWidgets.QLabel(Widget)
        self.train_label.setGeometry(QtCore.QRect(20, 210, 521, 16))
        
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.train_label.setFont(font)
        self.train_label.setObjectName("train_label")
        self.test_label = QtWidgets.QLabel(Widget)
        self.test_label.setGeometry(QtCore.QRect(20, 330, 521, 16))
        
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.test_label.setFont(font)
        self.test_label.setObjectName("test_label")
        self.hesapla = QtWidgets.QPushButton(Widget)
        self.hesapla.setGeometry(QtCore.QRect(160, 410, 131, 28))
        
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.hesapla.setFont(font)
        self.hesapla.setObjectName("hesapla")
        self.model_label = QtWidgets.QLabel(Widget)
        self.model_label.setGeometry(QtCore.QRect(20, 90, 521, 16))
        
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.model_label.setFont(font)
        self.model_label.setObjectName("model_label")
        self.label_3 = QtWidgets.QLabel(Widget)
        self.label_3.setGeometry(QtCore.QRect(20, 150, 141, 16))
        
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label = QtWidgets.QLabel(Widget)
        self.label.setGeometry(QtCore.QRect(20, 40, 131, 16))
        
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_6 = QtWidgets.QLabel(Widget)
        self.label_6.setGeometry(QtCore.QRect(20, 270, 141, 16))
        
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.gorsel_labeli = QtWidgets.QLabel(Widget)
        self.gorsel_labeli.setGeometry(QtCore.QRect(550, 50, 681, 712))
        self.gorsel_labeli.setText("")
        self.gorsel_labeli.setObjectName("gorsel_labeli")
        self.trainveriseti = QtWidgets.QPushButton(Widget)
        self.trainveriseti.setGeometry(QtCore.QRect(180, 150, 171, 28))
        
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.trainveriseti.setFont(font)
        self.trainveriseti.setObjectName("trainveriseti")
        self.modelsec = QtWidgets.QPushButton(Widget)
        self.modelsec.setGeometry(QtCore.QRect(170, 40, 131, 28))
        
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.modelsec.setFont(font)
        self.modelsec.setObjectName("modelsec")
        self.line_2 = QtWidgets.QFrame(Widget)
        self.line_2.setGeometry(QtCore.QRect(0, 240, 551, 20))
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.label_2 = QtWidgets.QLabel(Widget)
        self.label_2.setGeometry(QtCore.QRect(730, 10, 101, 21))
        
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.line = QtWidgets.QFrame(Widget)
        self.line.setGeometry(QtCore.QRect(0, 120, 551, 20))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.testveriseti = QtWidgets.QPushButton(Widget)
        self.testveriseti.setGeometry(QtCore.QRect(180, 270, 171, 28))
        
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.testveriseti.setFont(font)
        self.testveriseti.setObjectName("testveriseti")
        self.line_3 = QtWidgets.QFrame(Widget)
        self.line_3.setGeometry(QtCore.QRect(0, 360, 551, 20))
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        
        self.dogrulukorani = QtWidgets.QLabel(Widget)
        self.dogrulukorani.setGeometry(QtCore.QRect(300, 580, 311, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.dogrulukorani.setFont(font)
        self.dogrulukorani.setText("")
        self.dogrulukorani.setObjectName("dogrulukorani")

        self.retranslateUi(Widget)
        QtCore.QMetaObject.connectSlotsByName(Widget)

    def retranslateUi(self, Widget):
        _translate = QtCore.QCoreApplication.translate
        Widget.setWindowTitle(_translate("Widget", "Widget"))
        self.train_label.setText(_translate("Widget", "Train Veri Seti Adı"))
        self.test_label.setText(_translate("Widget", "Test Veri Seti Adı"))
        self.hesapla.setText(_translate("Widget", "Hesapla"))
        self.model_label.setText(_translate("Widget", "Model Dosya Adı"))
        self.label_3.setText(_translate("Widget", "Train Veri Seti :"))
        self.label.setText(_translate("Widget", "Model Seçimi :"))
        self.label_6.setText(_translate("Widget", "Test Veri Seti :"))
        self.trainveriseti.setText(_translate("Widget", "Train Veri Seti Seç"))
        self.modelsec.setText(_translate("Widget", "Model seç"))
        self.label_2.setText(_translate("Widget", "Çıktı"))
        self.testveriseti.setText(_translate("Widget", "Test Veri Seti Seç"))

