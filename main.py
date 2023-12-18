# run the UI file through pyuic5

import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
import matting
from ui import Ui_Dialog
import cv2
import numpy as np

class MyDialog(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self):
        super().__init__()
        self.fg_image = None
        self.trimap = None
        self.alpha_matte = None
        self.bg_image = None
        self.res_image = None

        self.matting_args = matting.args_init()
        self.matting_model = matting.model_load(self.matting_args)
        
        self.setupUi(self)
        self.select_fg.clicked.connect(self.select_foreground)
        self.select_tri.clicked.connect(self.select_trimap)  
        self.select_bg.clicked.connect(self.select_background)
        self.pre_alpha.clicked.connect(self.predict_alpha)
        self.com.clicked.connect(self.compose)

        self.save_tri.clicked.connect(self.save_trimap)
        self.save_alpha.clicked.connect(self.save_alpha_matte)
        self.save_com.clicked.connect(self.save_compose)


    def qimg2np(self, qimg):
        qimg = qimg.convertToFormat(QtGui.QImage.Format.Format_RGB888)
        width = qimg.width()
        height = qimg.height()
        ptr = qimg.bits()
        ptr.setsize(qimg.byteCount())
        arr = np.array(ptr).reshape(height, width, 3)
        return arr

    def np2qimg(self, arr):
        height, width, channel = arr.shape
        bytesPerLine = 3 * width
        qimg = QtGui.QImage(arr.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        return qimg

    def select_image(self):
        print("Select Foreground button clicked")
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)", options=options)
        if fileName:
            image = cv2.imread(fileName, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
            # pixmap = QtGui.QPixmap(fileName)
            # if not pixmap.isNull():
            #     image = pixmap.toImage()
            #     image = self.qimg2np(image)
            #     return image

    def show_image(self, image, graphicsView):
        pixmap = QtGui.QPixmap.fromImage(self.np2qimg(image))
        pixmap.scaled(graphicsView.size(), QtCore.Qt.KeepAspectRatio)
        scene = QtWidgets.QGraphicsScene()
        scene.addPixmap(pixmap)
        graphicsView.setScene(scene)
        graphicsView.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

    
    def select_foreground(self):
        self.fg_image = self.select_image()
        if self.fg_image is None:
            return
        self.show_image(self.fg_image, self.fg)

    def select_trimap(self):
        self.trimap = self.select_image()
        if self.trimap is None:
            return
        self.show_image(self.trimap, self.tri)

    def select_background(self):
        self.bg_image = self.select_image()
        if self.bg_image is None:
            return
        self.show_image(self.bg_image, self.bg)

    def predict_alpha(self):
        if self.fg_image is None or self.trimap is None:
            print("Foreground image or trimap is not selected")
            return
        self.alpha_matte = matting.matting(self.matting_args, self.matting_model, self.fg_image, self.trimap)
        self.show_image(self.alpha_matte, self.alpha)

    def compose(self):
        if self.fg_image is None or self.trimap is None:
            print("Foreground image or background image or trimap is not selected")
            return
        self.res_image = matting.composing(self.fg_image, self.alpha_matte, self.bg_image)
        self.show_image(self.res_image, self.com_res)

    
    def save_img(self, image):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)", options=options)
        if fileName:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(fileName, image)

    def save_trimap(self):
        if self.trimap is None:
            print("Trimap is not selected")
            return
        self.save_img(self.trimap)

    def save_alpha_matte(self):
        if self.alpha_matte is None:
            print("Alpha matte is not selected")
            return
        self.save_img(self.alpha_matte)

    def save_compose(self):
        if self.res_image is None:
            print("Compose image is not selected")
            return
        self.save_img(self.res_image)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    dialog = MyDialog()
    dialog.show()
    sys.exit(app.exec_())
