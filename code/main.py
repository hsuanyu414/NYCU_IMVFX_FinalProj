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
from seg import generater_trimap, segment

class MyDialog(QtWidgets.QDialog, Ui_Dialog):
    # inherit from both classes and add some attributes and links to buttons
    def __init__(self):
        super().__init__()
        # add attributes
        self.fg_image = None
        self.segment = None
        self.trimap = None
        self.alpha_matte = None
        self.pure_fg = None
        self.bg_image = None
        self.res_image = None
        self.size_value = 20
        self.defg_value = 0
        self.num_iters_value = 0
        self.matting_args = matting.args_init()
        self.matting_model = matting.model_load(self.matting_args)
        
        self.setupUi(self)
        
        # link actions to functions
        self.select_fg.clicked.connect(self.select_foreground)
        self.crop_fg.clicked.connect(self.crop_foreground)
        self.select_tri.clicked.connect(self.select_trimap) 
        self.gen_seg.clicked.connect(self.generater_segment)   
        self.gen_tri.clicked.connect(self.generater_trimap)  
        self.select_bg.clicked.connect(self.select_background)
        self.pre_alpha.clicked.connect(self.predict_alpha)
        self.com.clicked.connect(self.compose)
        self.spinBox_size.valueChanged.connect(self.size_changed)
        self.spinBox_num_iters.valueChanged.connect(self.num_iters_changed)
        self.comboBox.currentIndexChanged.connect(self.defg_changed)
        self.save_tri.clicked.connect(self.save_trimap)
        self.save_alpha.clicked.connect(self.save_alpha_matte)
        self.save_com.clicked.connect(self.save_compose)
        self.save_com_pure.clicked.connect(self.save_compose_pure)

    def qimg2np(self, qimg):
        # input qimg is a QImage object and output arr is a numpy array
        qimg = qimg.convertToFormat(QtGui.QImage.Format.Format_RGB888)
        width = qimg.width()
        height = qimg.height()
        ptr = qimg.bits()
        ptr.setsize(qimg.byteCount())
        arr = np.array(ptr).reshape(height, width, 3)
        return arr

    def np2qimg(self, arr):
        # input arr is a numpy array and output qimg is a QImage object
        height, width, channel = arr.shape
        bytesPerLine = 3 * width
        qimg = QtGui.QImage(arr.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        return qimg

    def select_image(self):
        # select image and return a numpy array
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)", options=options)
        if fileName:
            image = cv2.imread(fileName, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image

    def show_image(self, image, graphicsView):
        # show image in graphicsView
        pixmap = QtGui.QPixmap.fromImage(self.np2qimg(image))
        pixmap.scaled(graphicsView.size(), QtCore.Qt.KeepAspectRatio)
        scene = QtWidgets.QGraphicsScene()
        scene.addPixmap(pixmap)
        graphicsView.setScene(scene)
        graphicsView.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

    # select related functions
    def select_foreground(self):
        print("Select Foreground button clicked")
        ret = self.select_image()
        if ret is None:
            return
        else:
            self.fg_image = ret
        self.show_image(self.fg_image, self.fg)

    def select_trimap(self):
        print("Select Trimap button clicked")
        ret = self.select_image()
        if ret is None:
            return
        else:
            self.trimap = ret
        self.show_image(self.trimap, self.tri)

    def select_background(self):
        print("Select Background button clicked")
        ret = self.select_image()
        if ret is None:
            return
        else:
            self.bg_image = ret 
        self.show_image(self.bg_image, self.bg)
    # end of select related functions

    def generater_segment(self):
        print("Generate Segment button clicked")
        if self.fg_image is None:
            return
        self.segment = segment(self.fg_image)
        self.show_image(self.segment, self.seg)

    def size_changed(self):
        print("Unknown Region Thickness Changed")
        self.size_value = self.spinBox_size.value()
        #print(self.size_value)
    
    def defg_changed(self):
        print("Defg Changed")
        self.defg_value = self.comboBox.currentIndex()
        #print(self.defg_value)
    
    def num_iters_changed(self):
        print("Num Iters Changed")
        self.num_iters_value = self.spinBox_num_iters.value()
        #print(self.num_iters_value)
    
    def generater_trimap(self):
        print("Generate Trimap button clicked")
        if self.segment is None:
            return
        img = cv2.cvtColor(self.segment, cv2.COLOR_BGR2HSV)
        def getpos(event,x,y,flags,param):
            if event == 1:
                HSV_color = img[y, x]
                #print(f"x: {x}, y: {y}, hsv: {HSV_color}")
                if ~(HSV_color[0] == 0 and HSV_color[1] == 0 and HSV_color[1] == 0):
                    lower_bound = np.array([HSV_color[0], HSV_color[1], HSV_color[2]])
                    upper_bound = np.array([HSV_color[0], HSV_color[1], HSV_color[2]])
                    mask = cv2.inRange(img, lower_bound, upper_bound)
                    img[mask > 0] = [255, 255, 255]
                    cv2.imshow('Mask Image', img)

        cv2.imshow('Mask Image', img)
        cv2.setMouseCallback('Mask Image', getpos)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        lower_bound = np.array([0,0, 0])
        upper_bound = np.array([255, 255, 244])
        mask = cv2.inRange(img, lower_bound, upper_bound)
        img[mask > 0] = [0, 0, 0]
        mask_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.trimap = generater_trimap(mask_img,self.size_value,self.defg_value,self.num_iters_value)

        self.show_image(self.trimap, self.tri)


    # predict alpha matte
    def predict_alpha(self):
        if self.fg_image is None or self.trimap is None:
            print("Foreground image or trimap is not selected")
            return
        self.alpha_matte = matting.matting(self.matting_args, self.matting_model, self.fg_image, self.trimap)
        self.pure_fg = matting.composing(self.fg_image, self.alpha_matte)
        self.show_image(self.alpha_matte, self.alpha)

    # compose
    def compose(self):
        if self.fg_image is None or self.trimap is None:
            print("Foreground image or background image or trimap is not selected")
            return
        self.res_image = matting.composing(self.fg_image, self.alpha_matte, self.bg_image)
        self.show_image(self.res_image, self.com_res)
    
    # save image
    def save_img(self, image):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)", options=options)
        if fileName:
            # check if the file extension is correct
            if fileName.split(".")[-1] not in ["png", "jpg", "jpeg", "bmp", "gif"]:
                fileName += ".png"
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            elif image.shape[2] == 4: # transparent image (RGBA image)
                # check if the file extension is correct for transparent image
                if fileName.split(".")[-1] != "png":
                    fileName += ".png"
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
            else:
                print("Image shape is not correct")
                return
            cv2.imwrite(fileName, image)

    # save related functions
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

    def save_compose_pure(self):
        if self.pure_fg is None:
            print("Compose image is not selected")
            return
        self.save_img(self.pure_fg)
    # end of save related functions

    # crop foreground
    def crop_foreground(self):
        if self.fg_image is None:
            print("Foreground image is not selected")
            return
        fg_image_temp = self.fg_image.copy()
        fg_image_temp = cv2.cvtColor(fg_image_temp, cv2.COLOR_RGB2BGR)

        clone_image = fg_image_temp.copy() 
        cv2.namedWindow("Select Region to Crop")
        cv2.imshow("Select Region to Crop", clone_image)

        cropping = False
        x_start, y_start, x_end, y_end = 0, 0, 0, 0

        # define crop function for CV2 
        def crop(event, x, y, flags, param):
            nonlocal x_start, y_start, x_end, y_end, cropping, clone_image, fg_image_temp

            if event == cv2.EVENT_LBUTTONDOWN: # initialize the crop
                x_start, y_start, x_end, y_end = x, y, x, y
                cropping = True

            elif event == cv2.EVENT_MOUSEMOVE: # show the crop region when mouse moves
                if cropping:
                    clone_image = fg_image_temp.copy()
                    cv2.rectangle(clone_image, (x_start, y_start), (x, y), (0, 255, 0), 2)
                    cv2.imshow("Select Region to Crop", clone_image)

            elif event == cv2.EVENT_LBUTTONUP: # end up the crop and show the preview of cropped image
                x_end, y_end = x, y
                cropping = False
                cv2.rectangle(clone_image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
                cv2.imshow("Select Region to Crop", clone_image)

        cv2.setMouseCallback("Select Region to Crop", crop)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 13: # press enter to confirm
                break
        cv2.destroyAllWindows()

        # confirm the region to crop and show the cropped image then return
        cropped_image = fg_image_temp[min(y_start, y_end):max(y_start, y_end), min(x_start, x_end):max(x_start, x_end)]
        cv2.imshow("Cropped Foreground Image", cropped_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        self.fg_image = cropped_image
        self.show_image(self.fg_image, self.fg)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    dialog = MyDialog()
    dialog.show()
    sys.exit(app.exec_())
