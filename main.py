# -*- coding: utf-8 -*-
import sys 	
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import PyQt5
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget
from PyQt5 import QtCore, uic, QtWidgets
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMessageBox
from usedFunctions import *

from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWidgets import QApplication, QMainWindow


# resolution settings
if hasattr(QtCore.Qt,'AA_EnableHighDpiScaling'):
    PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling,True)
if hasattr(QtCore.Qt,'AA_UseHighDpiPixmaps'):
    PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps,True)

class PaintPicture(QDialog):
    
    ''' choose picture dialog'''

    def __init__(self, parent=None):
        super(PaintPicture, self).__init__()
        layout = QVBoxLayout()
        self.num = 0
        self.setWindowTitle('Please confirm')
        self.setLayout(layout)

    def showImage(self,filename):
        image = QImage(filename)
        self.imageLabel = QLabel()
        self.imageLabel.setPixmap(QPixmap.fromImage(image))
        layout = self.layout()       
        self.start = QPushButton('Start Decoding')
        self.rechoose = QPushButton('Rechoose Picture')
         # setting geometry of button
        self.start.setGeometry(200, 150, 100, 40)  
        # changing font and size of text
        self.start.setFont(QFont('Times', 15))
        self.rechoose.setGeometry(200, 150, 100, 40) 
        # changing font and size of text
        self.rechoose.setFont(QFont('Times', 15))
        self.layout2 = QHBoxLayout()
        self.layout2.addWidget(self.start)
        self.layout2.addWidget(self.rechoose)
        layout.addLayout(self.layout2)
        layout.addWidget(self.imageLabel)

class MyMainWindow(QtWidgets.QMainWindow, uic.loadUiType("dependencies/frmmain.ui")[0]):
    
    def __init__(self, parent=None):    
        super(MyMainWindow, self).__init__(parent)
        self.setupUi(self)
        widget = self.stackedWidget.widget(0)
        self.stackedWidget.setCurrentWidget(widget)
        self.scene = None 
        self.upc_1.clicked.connect(lambda: self.showpic(r'dependencies/images/070470409665.jpg'))
        self.upc_2.clicked.connect(lambda: self.showpic(r'dependencies/images/689076338486.jpg'))
        self.upc_3.clicked.connect(lambda: self.showpic(r'dependencies/images/014149929962.bmp'))
        self.code128_1.clicked.connect(lambda: self.showpic(r'dependencies/images/Code128 5mil 10chars 18cm.jpg'))
        self.code128_2.clicked.connect(lambda: self.showpic(r'dependencies/images/Code128 5mil 10chars 36cm.jpg'))
        self.code128_3.clicked.connect(lambda: self.showpic(r'dependencies/images/code128 5mil 10chars.jpg'))
        self.code39_1.clicked.connect(lambda: self.showpic(r'dependencies/images/code39 5mil 37cm.jpg'))
        self.code39_2.clicked.connect(lambda: self.showpic(r'dependencies/images/code39 5mil 18cm.jpg'))
        self.code39_3.clicked.connect(lambda: self.showpic(r'dependencies/images/code39 5mil 3chars.jpg'))
        self.setWindowFlags(QtCore.Qt.WindowMinimizeButtonHint | QtCore.Qt.WindowCloseButtonHint)  

    def showpic(self,imagename):
        self.imgshow = PaintPicture()
        self.imgshow.showImage(imagename)
        self.imgshow.show()
        self.imgshow.start.clicked.connect(lambda: self.start_decode(imagename))
        self.imgshow.rechoose.clicked.connect(self.imgshow.close)
    

    def start_decode(self,imagename):
        result = decode_image(imagename)
        self.show_html(imagename,result)

    def show_html(self,imagename,result):
        html = get_html(imagename,result)
        self.plot_widget = QWebEngineView()
        self.plot_widget.setWindowTitle('Result')
        self.plot_widget.setHtml(html)
        self.plot_widget.show()
        self.imgshow.close()
        
        
    
if __name__=="__main__":  
    app = QtWidgets.QApplication(sys.argv) 
    app.setWindowIcon(QIcon("ico.ico"))
    myWin = MyMainWindow()    
    myWin.show() 
    sys.exit(app.exec_())