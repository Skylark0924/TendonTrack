# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UI_Track.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Track(object):
    def setupUi(self, Track):
        Track.setObjectName("Track")
        Track.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(Track)
        self.centralwidget.setObjectName("centralwidget")
        self.ManualGroup = QtWidgets.QGroupBox(self.centralwidget)
        self.ManualGroup.setGeometry(QtCore.QRect(20, 40, 301, 201))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.ManualGroup.setFont(font)
        self.ManualGroup.setObjectName("ManualGroup")
        self.Left = QtWidgets.QPushButton(self.ManualGroup)
        self.Left.setGeometry(QtCore.QRect(40, 80, 70, 50))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.Left.setFont(font)
        self.Left.setObjectName("Left")
        self.Right = QtWidgets.QPushButton(self.ManualGroup)
        self.Right.setGeometry(QtCore.QRect(180, 80, 70, 50))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.Right.setFont(font)
        self.Right.setObjectName("Right")
        self.Down = QtWidgets.QPushButton(self.ManualGroup)
        self.Down.setGeometry(QtCore.QRect(110, 130, 70, 50))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.Down.setFont(font)
        self.Down.setObjectName("Down")
        self.Up = QtWidgets.QPushButton(self.ManualGroup)
        self.Up.setGeometry(QtCore.QRect(110, 30, 70, 50))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.Up.setFont(font)
        self.Up.setObjectName("Up")
        self.AutoGroup = QtWidgets.QGroupBox(self.centralwidget)
        self.AutoGroup.setGeometry(QtCore.QRect(440, 30, 251, 211))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.AutoGroup.setFont(font)
        self.AutoGroup.setObjectName("AutoGroup")
        self.Auto_move = QtWidgets.QPushButton(self.AutoGroup)
        self.Auto_move.setGeometry(QtCore.QRect(70, 100, 120, 50))
        self.Auto_move.setObjectName("Auto_move")
        self.ManualGroup_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.ManualGroup_2.setEnabled(True)
        self.ManualGroup_2.setGeometry(QtCore.QRect(20, 250, 301, 271))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.ManualGroup_2.setFont(font)
        self.ManualGroup_2.setObjectName("ManualGroup_2")
        self.Left_2 = QtWidgets.QPushButton(self.ManualGroup_2)
        self.Left_2.setGeometry(QtCore.QRect(40, 110, 70, 50))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.Left_2.setFont(font)
        self.Left_2.setObjectName("Left_2")
        self.Right_2 = QtWidgets.QPushButton(self.ManualGroup_2)
        self.Right_2.setGeometry(QtCore.QRect(180, 110, 70, 50))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.Right_2.setFont(font)
        self.Right_2.setObjectName("Right_2")
        self.Down_2 = QtWidgets.QPushButton(self.ManualGroup_2)
        self.Down_2.setGeometry(QtCore.QRect(110, 160, 70, 50))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.Down_2.setFont(font)
        self.Down_2.setObjectName("Down_2")
        self.Up_2 = QtWidgets.QPushButton(self.ManualGroup_2)
        self.Up_2.setGeometry(QtCore.QRect(110, 60, 70, 50))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.Up_2.setFont(font)
        self.Up_2.setObjectName("Up_2")
        self.State_log = QtWidgets.QTextBrowser(self.centralwidget)
        self.State_log.setGeometry(QtCore.QRect(440, 300, 256, 192))
        self.State_log.setObjectName("State_log")
        Track.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(Track)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 23))
        self.menubar.setObjectName("menubar")
        Track.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(Track)
        self.statusbar.setObjectName("statusbar")
        Track.setStatusBar(self.statusbar)

        self.retranslateUi(Track)
        QtCore.QMetaObject.connectSlotsByName(Track)

    def retranslateUi(self, Track):
        _translate = QtCore.QCoreApplication.translate
        Track.setWindowTitle(_translate("Track", "MainWindow"))
        self.ManualGroup.setTitle(_translate("Track", "Manual"))
        self.Left.setText(_translate("Track", "Left"))
        self.Right.setText(_translate("Track", "Right"))
        self.Down.setText(_translate("Track", "Down"))
        self.Up.setText(_translate("Track", "Up"))
        self.AutoGroup.setTitle(_translate("Track", "Auto"))
        self.Auto_move.setText(_translate("Track", "Auto move"))
        self.ManualGroup_2.setTitle(_translate("Track", "Manual"))
        self.Left_2.setText(_translate("Track", "Left"))
        self.Right_2.setText(_translate("Track", "Right"))
        self.Down_2.setText(_translate("Track", "Down"))
        self.Up_2.setText(_translate("Track", "Up"))
