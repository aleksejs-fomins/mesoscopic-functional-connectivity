###############################
# Import system libraries
###############################
import os, sys, locale
import json
import numpy as np
import matplotlib.pyplot as plt

from PyQt5 import QtGui, QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

import cv2

###############################
# Export Root Directory
###############################
thisdir = os.path.dirname(os.path.abspath(__file__))
# rootdir = os.path.dirname(thisdir)
# sys.path.append(rootdir)

#######################################
# Compile QT File
#######################################
ui_ui_path = os.path.join(thisdir, 'moviemakerapp.ui')
ui_py_path = os.path.join(thisdir, 'moviemakerapp.py')
qtCompilePrefStr = 'pyuic5 ' + ui_ui_path + ' -o ' + ui_py_path
print("Compiling QT GUI file", qtCompilePrefStr)
os.system(qtCompilePrefStr)

###############################
# Import local libraries
###############################
from codes.moviemaker.draggable import DraggableCircle
from codes.moviemaker.moviemakerapp import Ui_MovieMakerApp
import codes.moviemaker.qthelper as qthelper

from codes.lib.data_io.matlab_lib import loadmat
from codes.lib.data_io.yaro.yaro_data_read import read_neuro_perf


'''
GUI for making functional connectivity videos
   
===================
= Current features
===================


===================
= Controls
===================
    +/-         Changes app font size


===================
= TODO
===================
[ ] Load FC datafile, extract matrix
[+] Load metadatafile, extract region labels, data traces
[ ] Store recent data in JSON

[ ] Circles :: Represent Regions
    [ ] Extract from datafile
    [ ] Enable variable radius
    [ ] Label on plot
    [ ] Optional brightness based on data trace
    [ ] Enable drag
    [ ] Enable save/load position based on index
    [ ] Disable individual regions

[ ] Edges :: Represent Connections
    [ ] Visible based on P < 0.01
      [ ] Optional 2 directed edges based on P 
      [ ] Optional undirected edge based on (P + P^T)/2
    [ ] Optional edge thickness from TE
    
[ ] Movie:
  [ ] Slider to select min-max timestep
  [ ] Preview in app
  [ ] Save to movie file
'''

#######################################################
# Main Window
#######################################################
class MovieMakerApp():
    def __init__(self, dialog):
        # Init
        self.dialog = dialog
        self.gui = Ui_MovieMakerApp()
        self.gui.setupUi(dialog)

        # GUI-Constants
        self.fontsize = 15
        self.CIRCLE_RADIUS_DEFAULT = 5
        self.AUTOGEN_CIRCLE_RADIUS = 100

        # Load settings file
        with open('settings.json', 'r') as json_file:
            settingsDict = json.load(json_file)
            self.pathNeuro  = settingsDict["pathNeuro"]
            self.pathFC     = settingsDict["pathFC"]
            self.pathLayout = settingsDict["pathLayout"]
            self.pathMovie  = settingsDict["pathMovie"]

        self.globalStateDict = {
            "HAVE_NODES"        : False,
            "HAVE_DATA_NEURO"   : False,
            "HAVE_DATA_FC"      : False,
            "HAVE_NODE_COORDS"  : False
        }

        # Init Plot Layout
        self.init_plot_layout()

        # Listeners - UI
        self.dialog.keyPressEvent = self.keyPressEvent
        self.gui.actionLoad_Data_Folder.triggered.connect(self.load_data_folder)
        self.gui.actionLoad_FC_File.triggered.connect(self.load_fc_folder)
        self.gui.actionLoad_node_layout.triggered.connect(self.load_node_layout)
        self.gui.actionSave_node_layout.triggered.connect(self.save_node_layout)
        self.gui.actionLoad_Recent.triggered.connect(self.load_recent)
        self.gui.actionExport_movie.triggered.connect(self.export_movie)

        self.gui.actionUpdate_Plot.triggered.connect(self.update_plot)
        self.gui.actionAutogen_Circles.triggered.connect(self.autogen_node_layout)

        # self.gui.movieUpdateButton.clicked.connect(self.update_plot())
        # self.gui.movieAutogenButton.clicked.connect(self.autogen_node_layout)
        self.gui.movieFrameSlider.valueChanged.connect(self.movie_slider_react)


    def init_plot_layout(self):
        # Setup plotting
        self.movieFigure, self.movieAxis = plt.subplots()
        self.movieAxis.axis('off')

        self.movieFigureCanvas = FigureCanvas(self.movieFigure)
        self.movieFigureToolbar = NavigationToolbar(self.movieFigureCanvas, self.dialog)
        self.gui.movieFrameLayout.addWidget(self.movieFigureToolbar)
        self.gui.movieFrameLayout.addWidget(self.movieFigureCanvas)


    def load_data_folder(self, cached=True):
        # Get path using gui, or use cached path
        if not cached:
            pathTmp = self.pathNeuro if self.pathNeuro is not None else "./"
            self.pathNeuro = QtWidgets.QFileDialog.getExistingDirectory(self.gui.centralWidget, "Load Neuronal Data Directory", pathTmp)

        # Read neuro data
        self.dataNeuro, self.behavior, self.performance = read_neuro_perf(self.pathNeuro, verbose=True)
        self.nTrial, self.nTime, self.nChannel = self.dataNeuro.shape

        # Read channel labels
        pathNeuroParent = os.path.dirname(self.pathNeuro)
        pathChannelLabels = os.path.join(pathNeuroParent, 'channel_labels.mat')
        if os.path.isfile(pathChannelLabels):
            self.channelLabels = loadmat(pathChannelLabels)['channel_labels']
        else:
            print('Warning: Channel Label file not found, using number labels instead')
            self.channelLabels = np.array([str(i) for i in range(self.nChannel)])

        # Fill channel table
        self.gui.channelTableWidget.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.gui.channelTableWidget.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.gui.channelTableWidget.setRowCount(0)
        for iRow, label in enumerate(self.channelLabels):
            thisCheckBox = qthelper.qtable_make_checkbox(checked=True)
            qthelper.qtable_addrow(self.gui.channelTableWidget, iRow, [thisCheckBox] + [label])

        # Update global state
        self.globalStateDict["HAVE_NODES"] = True
        self.globalStateDict["HAVE_DATA_NEURO"] = True

        # If everything went smooth, save path to settings file
        self.save_settings()


    def load_fc_folder(self, cached=True):
        # Get path using gui, or use cached path
        if not cached:
            pathTmp = self.pathFC if self.pathFC is not None else "./"
            self.pathFC = QtWidgets.QFileDialog.getExistingDirectory(self.gui.centralWidget, "Load FC Directory", pathTmp)

        # Extract expected filename from neuro path
        # Open that file
        # Extract TE mat and store locally


    def load_node_layout(self, cached=True):
        # Get path using gui, or use cached path
        if not cached:
            pathTmp = os.path.dirname(self.pathLayout) if self.pathLayout is not None else "./"
            self.pathLayout = QtWidgets.QFileDialog.getOpenFileName(self.gui.centralWidget, "Load Node Layout", pathTmp, "JSON Files (*.json)")

            # Load settings file
            with open(self.pathLayout, 'r') as json_file:
                nodeDict = json.load(json_file)

            channelEnabledDictNew = {key : val[0] for key, val in nodeDict.items()}
            self.channelCoordsDict = {key : val[1,2] for key, val in nodeDict.items()}

            self.update_channel_table(channelEnabledDictNew)

            # We have coordinates only if all coordinates are not None
            haveCoordX = [v[0] is not None for v in self.channelCoordsDict.values()]
            haveCoordY = [v[1] is not None for v in self.channelCoordsDict.values()]
            self.globalStateDict["HAVE_NODE_COORDS"] = np.all(haveCoordX) and np.all(haveCoordY)


    def save_node_layout(self):
        if not self.globalStateDict["HAVE_NODES"]:
            print("There are no nodes to save")
        else:
            pathTmp = os.path.dirname(self.pathLayout) if self.pathLayout is not None else "./"
            self.pathLayout = QtWidgets.QFileDialog.getSaveFileName(self.gui.centralWidget, "Save Node Layout", pathTmp, "JSON Files (*.json)")

            channelEnabledDict = self.parse_channel_table()

            if not self.globalStateDict["HAVE_NODE_COORDS"]:
                self.channelCoordsDict = {key : [None, None] for key in channelEnabledDict.keys()}

            nodeDict = {label : [checked] + self.channelCoordsDict[label] for label,checked in channelEnabledDict.items()}

            with open(self.pathLayout, 'w') as f:
                json.dump(nodeDict, f, indent=4)
            print("Saved node layout to", self.pathLayout )


    def save_settings(self):
        settingsDict = {
            "pathNeuro"     : self.pathNeuro,
            "pathFC"        : self.pathFC,
            "pathLayout"    : self.pathLayout,
            "pathMovie"     : self.pathMovie
        }
        with open('settings.json', 'w') as f:
            json.dump(settingsDict, f, indent=4)
        print("Saved settings file")


    # Load most recently-used data files using paths from local settings file
    def load_recent(self):
        def try_load(loadfunc, path, altstr):
            if path is not None:
                loadfunc(cached=True)
            else:
                print(altstr)

        try_load(self.load_data_folder,   self.pathNeuro,  "Path to neuronal data not initialized, skipping")
        try_load(self.load_fc_folder,     self.pathFC,     "Path to FC data not initialized, skipping")
        try_load(self.load_node_layout(), self.pathLayout, "Path to node layout file not initialized, skipping")


    def export_movie(self):
        pathTmp = os.path.dirname(self.pathMovie) if self.pathMovie is not None else "./"
        self.pathNodeLayoutFile = QtWidgets.QFileDialog.getSaveFileName(self.gui.centralWidget, "Save Movie File", pathTmp, "Movie Files (*.avi)")


    # Extract values from channel table by column
    def parse_channel_table(self):
        channelEnabled = qthelper.qtable_getcolchecked(self.gui.channelTableWidget, 0)
        channelLabels  = qthelper.qtable_getcoltext(self.gui.channelTableWidget, 1)
        return dict(zip(channelLabels, channelEnabled))


    # Updates checked states from file
    def update_channel_table(self, channelEnabledDictNew):
        self.channelEnabledDict = self.parse_channel_table()
        for k,v in channelEnabledDictNew.items():
            self.channelEnabledDict[k] = v
        qthelper.qtable_setcolchecked(self.gui.channelTableWidget, 0, list(self.channelEnabledDict.values()))


    # Arrange nodes on a circle
    def autogen_node_layout(self):
        self.channelEnabledDict = self.parse_channel_table()
        self.channelCoordsDict = {}

        dAlpha = 2 * np.pi / self.nChannel
        for i, key in enumerate(self.channelEnabledDict.keys()):
            x = self.AUTOGEN_CIRCLE_RADIUS * np.cos(i * dAlpha)
            y = self.AUTOGEN_CIRCLE_RADIUS * np.sin(i * dAlpha)
            self.channelCoordsDict[key] = [x, y]

        self.plot_circles()
        self.globalStateDict["HAVE_NODE_COORDS"] = True


    def plot_circles(self, iTime=None, draggable=False, update=False):
        if not update:
            self.movieCircles = []
            for x, y in self.channelCoordsDict.values():
                self.movieCircles += [plt.Circle((x, y), self.CIRCLE_RADIUS_DEFAULT, color='blue')]
                self.movieAxis.add_artist(self.movieCircles[-1])
            self.movieAxis.set_xlim(-1.5 * self.AUTOGEN_CIRCLE_RADIUS, 1.5 * self.AUTOGEN_CIRCLE_RADIUS)
            self.movieAxis.set_ylim(-1.5 * self.AUTOGEN_CIRCLE_RADIUS, 1.5 * self.AUTOGEN_CIRCLE_RADIUS)
            self.movieFigureCanvas.draw()
        else:
            pass


    def plot_edges(self, iTime=None, update=False):
        pass


    def movie_slider_react(self):
        thisFrame = self.gui.movieFrameSlider.value()
        self.gui.movieFrameLabel.setText(str(thisFrame))


    def update_plot(self, iTime):
        self.plot_circles(iTime)
        self.plot_edges(iTime)


    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key_Plus or e.key() == QtCore.Qt.Key_Minus:
            self.fontsize += int(e.key() == QtCore.Qt.Key_Plus) - int(e.key() == QtCore.Qt.Key_Minus)
            print("New font size", self.fontsize)
            self.gui.centralWidget.setStyleSheet("font-size: " + str(self.fontsize) + "pt;")
            self.gui.menuBar.setStyleSheet("font-size: " + str(self.fontsize) + "pt;")


######################################################
# Start the QT window
######################################################
if __name__ == '__main__' :
    app = QtWidgets.QApplication(sys.argv)
    mainwindow = QtWidgets.QMainWindow()
    locale.setlocale(locale.LC_TIME, "en_GB.utf8")
    pth1 = MovieMakerApp(mainwindow)
    mainwindow.show()
    sys.exit(app.exec_())
