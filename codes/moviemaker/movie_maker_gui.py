###############################
# Import system libraries
###############################
import os, sys, locale
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

###############################
# Export Root Directory
###############################
thisdir = os.path.dirname(os.path.abspath(__file__))
rootdir = os.path.dirname(os.path.dirname(thisdir))
sys.path.append(rootdir)

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
from codes.moviemaker.opencv_lib import cvWriter
from codes.moviemaker.matplotlib_helper import fig2numpy, get_fig_size_pixels

from codes.lib.data_io.os_lib import getfiles_walk
from codes.lib.data_io.matlab_lib import loadmat
from codes.lib.data_io.yaro.yaro_data_read import read_neuro_perf, readTE_H5, parse_TE_folder

import codes.lib.stat.graph_lib as graph_lib


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

[ ] Core        ::: MAKE SURE EDGE DIRECTIONS ARE NOT INVERTED
[ ] Core        ::: MAKE SURE TIME ON SLIDER MATCHES ACTUAL TIME OF LINKS
[ ] Core        ::: Synchronize data and FC time. Make reader return both shifts along with dataset

[ ] UI          ::: Solve Keyboard KeyEvent not activaing
[ ] UI          ::: React combo box ::: restart slider; reload img
[ ] UI          ::: Autogen circles ::: If have FC, auto-update edges based on FC
[ ] UI          ::: Tickbox-based ::: If have FC, auto-update edges based on FC
[ ] UI          ::: Impl Settings Tab

[ ] Performance ::: Can optimize write-time by optional disabling canvas.draw? 
[ ] Tidy Code   ::: Move most dictionaries from Key to integer value. Only keep dict for interacting with table

[ ] Extension   ::: Allow edit circle names, react accordingly
[ ] Extension   ::: Circle radius based on IN/OutDegree
[ ] Extension   ::: Circle brightness based on data trace
[ ] Extension   ::: Undirected edges based on (P + P^T)/2
[ ] Extension   ::: Settings ::: Crop time window [min/max in seconds]
[ ] Extension   ::: Settings ::: Timestamp on movie
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
            "HAVE_NODE_COORDS"  : False,
            "HAVE_EDGES"        : False
        }
        self.react_global_state_change()

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
        self.gui.actionAutogen_Circles.triggered.connect(self.autogen_node_layout)

        self.gui.movieFrameSlider.valueChanged.connect(self.movie_slider_react)
        self.gui.fcDataComboBox.currentIndexChanged.connect(self.load_fc_data)


    def init_plot_layout(self):
        # Setup plotting
        # self.movieFigSize = (6, 6)
        # self.movieFigDPI = 100
        # self.movieFigure, self.movieAxis = plt.subplots(figsize=self.movieFigSize, dpi=self.movieFigDPI)
        self.movieFigure, self.movieAxis = plt.subplots()
        self.movieAxis.axis('off')

        self.movieFigureCanvas = FigureCanvas(self.movieFigure)
        self.movieFigureToolbar = NavigationToolbar(self.movieFigureCanvas, self.dialog)
        self.gui.movieFrameLayout.addWidget(self.movieFigureToolbar)
        self.gui.movieFrameLayout.addWidget(self.movieFigureCanvas)


    def set_global_state(self, key, val):
        self.globalStateDict[key] = val
        print("Global State of", key, "updated to", val)
        self.react_global_state_change()


    #######################################################
    # Reacting to GUI elements
    #######################################################


    # Enable/Disable actions based on global state
    def react_global_state_change(self):
        self.gui.actionLoad_FC_File.setEnabled(self.globalStateDict["HAVE_DATA_NEURO"])
        self.gui.actionLoad_node_layout.setEnabled(self.globalStateDict["HAVE_NODES"])
        self.gui.actionSave_node_layout.setEnabled(self.globalStateDict["HAVE_NODE_COORDS"])
        self.gui.actionExport_movie.setEnabled(self.globalStateDict["HAVE_EDGES"])
        self.gui.actionAutogen_Circles.setEnabled(self.globalStateDict["HAVE_NODES"])
        self.gui.fcDataComboBox.setEnabled(self.globalStateDict["HAVE_DATA_FC"])
        self.gui.movieFrameSlider.setEnabled(self.globalStateDict["HAVE_EDGES"])


    def react_channel_table_itemchanged(self, item):
        iRow = item.row()
        iCol = item.column()

        if (iCol == 0) and (self.globalStateDict["HAVE_NODE_COORDS"]):
            nodeIdx = int(self.gui.channelTableWidget.item(iRow, 1).text())
            nodeLabel = self.gui.channelTableWidget.item(iRow, 2).text()
            self.update_circle_and_edges((nodeIdx, nodeLabel), circleMoved=False)


    def load_data_folder(self, cached=False):
        # Get path using gui, or use cached path
        if not cached:
            pathTmp = self.pathNeuro if self.pathNeuro is not None else "./"
            self.pathNeuro = QtWidgets.QFileDialog.getExistingDirectory(self.gui.centralWidget, "Load Neuronal Data Directory", pathTmp)

        if self.pathNeuro == "":
            print("No directory selected, skipping")
            self.pathNeuro = None
            return

        # Read neuro data
        self.dataNeuro, self.behavior, self.performance = read_neuro_perf(self.pathNeuro, verbose=True)
        self.nTrial, self.nTime, self.nChannel = self.dataNeuro.shape

        # Read channel labels
        pathNeuroParent = os.path.dirname(self.pathNeuro)
        pathChannelLabels = os.path.join(pathNeuroParent, 'channel_labels.mat')
        if os.path.isfile(pathChannelLabels):
            self.channelKeys = loadmat(pathChannelLabels)['channel_labels']
            self.channelKeys = [(i, v) for i, v in enumerate(self.channelKeys)]
        else:
            print('Warning: Channel Label file not found, using number labels instead')
            self.channelKeys = [(i, str(i)) for i in range(self.nChannel)]

        # Fill channel table
        self.gui.channelTableWidget.disconnect()  # Ensure table does not emit any user-signals before editing it (could happen when we load 2nd time)
        self.gui.channelTableWidget.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.gui.channelTableWidget.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.gui.channelTableWidget.setRowCount(0)
        for iRow, label in self.channelKeys:
            striRow = str(iRow) if iRow > 9 else '0' + str(iRow)
            thisCheckBox = qthelper.qtable_make_checkbox(checked=True)
            qthelper.qtable_addrow(self.gui.channelTableWidget, iRow, [thisCheckBox, striRow, label])

        qthelper.qtable_setcolflag(self.gui.channelTableWidget, 0, QtCore.Qt.ItemIsEditable, True)
        qthelper.qtable_setcolflag(self.gui.channelTableWidget, 1, QtCore.Qt.ItemIsEditable, False)
        qthelper.qtable_setcolflag(self.gui.channelTableWidget, 2, QtCore.Qt.ItemIsEditable, False)
        self.gui.channelTableWidget.itemChanged.connect(self.react_channel_table_itemchanged)

        # Update global state
        self.set_global_state("HAVE_NODES", True)
        self.set_global_state("HAVE_DATA_NEURO", True)

        # If everything went smooth, save path to settings file
        self.save_settings()


    def load_fc_folder(self, cached=False):
        if not self.globalStateDict["HAVE_DATA_NEURO"]:
            print("Can't load FC data before loading Neuro data, skipping")
        else:
            # Get path using gui, or use cached path
            if not cached:
                pathTmp = self.pathFC if self.pathFC is not None else "./"
                self.pathFC = QtWidgets.QFileDialog.getExistingDirectory(self.gui.centralWidget, "Load FC Directory", pathTmp)

            if self.pathFC == "":
                print("No directory selected, skipping")
                self.pathFC = None
                return

            # Extract summary from FC path
            self.summaryFC = parse_TE_folder(self.pathFC)

            # Extract all filenames that match the neuronal dataset
            mousekey = os.path.basename(self.pathNeuro)
            fileswalk = getfiles_walk(self.pathFC, [mousekey, ".h5"])
            self.pathFCFilesDict = {name : os.path.join(path, name) for path, name in fileswalk}
            if len(self.pathFCFilesDict) == 0:
                raise ValueError("Did not find any H5 files with key", mousekey)

            # Populate dataset combo box
            self.update_fc_combo_box()

            # Load FC data
            self.load_fc_data()

            # Update slider to FC values
            oldState = self.gui.movieFrameSlider.blockSignals(True)
            self.gui.movieFrameSlider.setValue(0)
            self.gui.movieFrameSlider.setMaximum(len(self.timesFC))
            self.gui.movieFrameSlider.blockSignals(oldState)

            # If everything went smooth, save path to settings file
            self.save_settings()

            # Update global state
            self.set_global_state("HAVE_DATA_FC", True)


    # Loads FC data corresponding to the current selection of combo box
    def load_fc_data(self):
        fcDataKey = self.gui.fcDataComboBox.currentText()
        fcDataPath = self.pathFCFilesDict[fcDataKey]
        self.timesFC, (self.fcMag, self.fcLag, self.fcPval) = readTE_H5(fcDataPath, self.summaryFC)
        self.fcIsConn = graph_lib.is_conn(self.fcPval, 0.01)
        print("loaded data for", fcDataKey)


    def load_node_layout(self, cached=False):
        # Get path using gui, or use cached path
        if not cached:
            pathTmp = os.path.dirname(self.pathLayout) if self.pathLayout is not None else "./"
            self.pathLayout = QtWidgets.QFileDialog.getOpenFileName(self.gui.centralWidget, "Load Node Layout", pathTmp, "JSON Files (*.json)")[0]

        if self.pathLayout == "":
            print("No directory selected, skipping")
            self.pathLayout = None
            return

        def unserialize_key(keyStr):
            k1, k2 = keyStr.split("&")
            return (int(k1), k2)

        # Load settings file
        with open(self.pathLayout, 'r') as json_file:
            nodeDict = json.load(json_file)
            nodeDict = {unserialize_key(k) : v for k,v in nodeDict.items()}

        channelEnabledDict = {key : val[0] for key, val in nodeDict.items()}
        channelCoordsDict = {key : val[1:3] for key, val in nodeDict.items()}

        # We have coordinates only if all coordinates are not None
        haveCoordX = [v[0] is not None for v in channelCoordsDict.values()]
        haveCoordY = [v[1] is not None for v in channelCoordsDict.values()]
        gotNodeCoords = np.all(haveCoordX) and np.all(haveCoordY)

        self.update_channel_table(channelEnabledDict)

        if gotNodeCoords:
            if not self.globalStateDict["HAVE_NODE_COORDS"]:
                self.plot_circles(channelCoordsDict)
                self.set_global_state("HAVE_NODE_COORDS", True)
            else:
                self.update_plot(channelCoordsDict=channelCoordsDict)
            self.save_settings()
        else:
            self.autogen_node_layout()


    def save_node_layout(self):
        if not self.globalStateDict["HAVE_NODES"]:
            print("There are no nodes to save")
        else:
            pathTmp = os.path.dirname(self.pathLayout) if self.pathLayout is not None else "./"
            self.pathLayout = QtWidgets.QFileDialog.getSaveFileName(self.gui.centralWidget, "Save Node Layout", pathTmp, "JSON Files (*.json)")[0]

            if self.pathLayout == "":
                print("No directory selected, skipping")
                self.pathLayout = None
                return

            channelEnabledDict = self.parse_channel_table()
            if self.globalStateDict["HAVE_NODE_COORDS"]:
                channelCoordsDict = self.get_circle_pos_from_plot()
            else:
                channelCoordsDict = {key: [None, None] for key in channelEnabledDict.keys()}

            nodeDict = {str(k[0]) + '&' + k[1] : [checked] + channelCoordsDict[k] for k,checked in channelEnabledDict.items()}

            with open(self.pathLayout, 'w') as f:
                json.dump(nodeDict, f, indent=4)
            print("Saved node layout to", self.pathLayout)


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
        try_load(self.load_node_layout,   self.pathLayout, "Path to node layout file not initialized, skipping")


    # TODO: Find out why color-swap ::: matplotlib or openCV. Move adjustment into either to matplotlibhelper or cv_wrapper
    def export_movie(self):
        pathTmp = os.path.dirname(self.pathMovie) if self.pathMovie is not None else "./"
        self.pathNodeLayoutFile = QtWidgets.QFileDialog.getSaveFileName(self.gui.centralWidget, "Save Movie File", pathTmp, "Movie Files (*.avi)")[0]
        if self.pathNodeLayoutFile == "":
            print("No file selected, skipping")
            return

        figSize = get_fig_size_pixels(self.movieFigure)
        figSize = (figSize[1], figSize[0])

        print("Started writing movie of shape", figSize, "to", self.pathNodeLayoutFile)
        self.gui.centralWidget.setEnabled(False)

        with cvWriter(self.pathNodeLayoutFile, figSize, frate=4, isColor=True) as movieWriter:
            for i in range(len(self.timesFC)):
                self.gui.movieFrameSlider.setValue(i)
                thisFig = fig2numpy(self.movieFigure)[:, :, [2,1,0]]
                print("Writing frame", i, "of shape", thisFig.shape)
                movieWriter.write(thisFig)

        self.gui.centralWidget.setEnabled(True)
        print("Finished writing movie")


    def movie_slider_react(self):
        thisFrame = self.gui.movieFrameSlider.value()
        self.gui.movieFrameLabel.setText(str(np.round(self.timesFC[thisFrame], 2)))
        self.update_plot()


    # Arrange nodes on a circle
    def autogen_node_layout(self):
        if not self.globalStateDict["HAVE_NODES"]:
            print("Can't generate node locations if there are no nodes")
        else:
            channelEnabledDict = self.parse_channel_table()
            channelCoordsDict = {}

            dAlpha = 2 * np.pi / self.nChannel
            for i, key in enumerate(channelEnabledDict.keys()):
                x = self.AUTOGEN_CIRCLE_RADIUS * np.cos(i * dAlpha)
                y = self.AUTOGEN_CIRCLE_RADIUS * np.sin(i * dAlpha)
                channelCoordsDict[key] = [x, y]

            self.plot_circles(channelCoordsDict)
            self.set_global_state("HAVE_NODE_COORDS", True)
            self.plot_edges()


    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key_Plus or e.key() == QtCore.Qt.Key_Minus:
            self.fontsize += int(e.key() == QtCore.Qt.Key_Plus) - int(e.key() == QtCore.Qt.Key_Minus)
            print("New font size", self.fontsize)
            self.gui.centralWidget.setStyleSheet("font-size: " + str(self.fontsize) + "pt;")
            self.gui.menuBar.setStyleSheet("font-size: " + str(self.fontsize) + "pt;")


    #######################################################
    # Auxiliary GUI routines
    #######################################################


    # Extract values from channel table by column
    def parse_channel_table(self):
        channelEnabled = qthelper.qtable_getcolchecked(self.gui.channelTableWidget, 0)
        channelIdxs    = qthelper.qtable_getcoltext(self.gui.channelTableWidget, 1)
        channelLabels  = qthelper.qtable_getcoltext(self.gui.channelTableWidget, 2)
        channelIdxs = [int(idx) for idx in channelIdxs]
        channelKeys = zip(channelIdxs, channelLabels)

        return dict(zip(channelKeys, channelEnabled))


    def get_circle_pos_from_plot(self):
        channelEnabledDict = self.parse_channel_table()
        return {key: list(self.movieCircles[key[0]].center) for key in channelEnabledDict.keys()}


    # Updates checked states from file
    def update_channel_table(self, channelEnabledDictNew):
        # Ensure no signals are passed during this operation
        oldState = self.gui.channelTableWidget.blockSignals(True)

        channelEnabledDict = self.parse_channel_table()
        for k,v in channelEnabledDictNew.items():
            if k in channelEnabledDict.keys():
                channelEnabledDict[k] = v
            else:
                print("Warning: key", k, "is not part of original channel keys")
        qthelper.qtable_setcolchecked(self.gui.channelTableWidget, 0, list(channelEnabledDict.values()))

        self.gui.channelTableWidget.blockSignals(oldState)


    # Fill FC combo box with names of available datasets
    def update_fc_combo_box(self):
        # Ensure no signals are passed during this operation
        oldState = self.gui.fcDataComboBox.blockSignals(True)
        qthelper.qcombo_fill(self.gui.fcDataComboBox, self.pathFCFilesDict.keys())
        self.gui.fcDataComboBox.blockSignals(oldState)


    #######################################################
    # Auxiliary plotting routines
    #######################################################


    def plot_circles(self, channelCoordsDict):
        self.movieCircles = []
        self.movieCirclesDraggable = []
        self.movieLabels = []

        channelEnabledDict = self.parse_channel_table()
        for key, pos in channelCoordsDict.items():
            thisVisible = channelEnabledDict[key]
            dragUpdate = lambda pos, key=key: self.update_circle_and_edges(key, circleMoved=True)
            self.movieCircles += [plt.Circle(pos, self.CIRCLE_RADIUS_DEFAULT, color='blue', visible=thisVisible)]
            self.movieAxis.add_artist(self.movieCircles[-1])
            self.movieCirclesDraggable += [DraggableCircle(self.movieCircles[-1], triggerOnRelease=dragUpdate)]

            labelPos = self.label_calc_rel_pos(pos)
            self.movieLabels += [self.movieAxis.text(labelPos[0], labelPos[1], key[1], visible=thisVisible)]

        self.movieAxis.set_xlim(-1.5 * self.AUTOGEN_CIRCLE_RADIUS, 1.5 * self.AUTOGEN_CIRCLE_RADIUS)
        self.movieAxis.set_ylim(-1.5 * self.AUTOGEN_CIRCLE_RADIUS, 1.5 * self.AUTOGEN_CIRCLE_RADIUS)
        self.movieFigureCanvas.draw()


    def plot_edges(self, iTime=None):
        self.movieEdgesIdxMap = {}
        self.movieEdges = []
        self.movieEdgePatches = []

        def make_edge(i, j):
            self.movieEdgesIdxMap[(i, j)] = len(self.movieEdges)
            self.movieEdges += [FancyArrowPatch((0, 0), (1, 1))]
            self.movieEdgePatches += [self.movieAxis.add_patch(self.movieEdges[-1])]
            self.update_edge(i,j,iTime=iTime)

        for i in range(self.nChannel):
            for j in range(i+1, self.nChannel):
                make_edge(i, j)
                make_edge(j, i)

        self.set_global_state("HAVE_EDGES", True)
        self.movieFigureCanvas.draw()


    def update_circle(self, nodeKey, enabled=None, newCoord=None, iTime=None):
        idx = nodeKey[0]
        circle = self.movieCircles[idx]
        label = self.movieLabels[idx]

        if enabled is None:
            enabled = self.parse_channel_table()[nodeKey]

        if newCoord is not None:
            print("Node", nodeKey, "auto-moved to", newCoord)
            circle.center = newCoord

        circle.set_visible(enabled)
        label.set_visible(enabled)
        label.set_position(self.label_calc_rel_pos(circle.center))


    def update_circle_and_edges(self, nodeKey, enabled=None, circleMoved=True):
        iTime = self.gui.movieFrameSlider.value() if self.globalStateDict["HAVE_DATA_FC"] else None

        self.update_circle(nodeKey, enabled=enabled, newCoord=None, iTime=iTime)

        # Update all edges connected to this node if they
        for (i, j), idxEdge in self.movieEdgesIdxMap.items():
            if (i == nodeKey[0]) or (j == nodeKey[0]):
                self.update_edge(i, j, circleMoved=circleMoved, iTime=iTime)

        self.movieFigureCanvas.draw()


    def update_edge(self, i, j, circleMoved=True, iTime=None):
        haveTime = iTime is not None

        idxEdge = self.movieEdgesIdxMap[(i,j)]
        vis1 = self.movieCircles[i].get_visible()
        vis2 = self.movieCircles[j].get_visible()
        visP = self.fcIsConn[i, j, iTime] if haveTime else True
        visEdge = vis1 and vis2 and visP
        alphaEdge = self.edge_calc_alpha_from_fc(self.fcMag[i,j, iTime]) if haveTime and visEdge else 1

        if circleMoved:
            p1 = np.array(self.movieCircles[i].center)
            p2 = np.array(self.movieCircles[j].center)
            p1sh, p2sh = self.edge_calc_rel_pos(p1, p2)

            posPrim = [p1sh, 0.03 * p1sh + 0.97 * p2sh]
            kw = dict(arrowstyle="Simple,tail_width=1,head_width=8,head_length=8", color="red", alpha=alphaEdge, visible=visEdge)

            self.movieEdgePatches[idxEdge].remove()
            self.movieEdges[idxEdge] = FancyArrowPatch(posPrim[0], posPrim[1], connectionstyle="arc3,rad=.2", **kw)
            self.movieEdgePatches[idxEdge] = self.movieAxis.add_patch(self.movieEdges[idxEdge])

        self.movieEdges[idxEdge].set_visible(visEdge)


    def update_plot(self, channelCoordsDict=None):
        iTime = self.gui.movieFrameSlider.value() if self.globalStateDict["HAVE_DATA_FC"] else None

        # Update all circles
        channelEnabledDict = self.parse_channel_table()
        for k,v in channelEnabledDict.items():
            newCoord = channelCoordsDict[k] if channelCoordsDict is not None else None
            self.update_circle(k, enabled=v, newCoord=newCoord, iTime=iTime)

        if self.globalStateDict["HAVE_EDGES"]:
            for (i, j), idxEdge in self.movieEdgesIdxMap.items():
                self.update_edge(i, j, iTime=iTime)
        else:
            self.plot_edges(iTime=iTime)

        self.movieFigureCanvas.draw()


    # Shift label w.r.t center of circle, so they don't overlap
    def label_calc_rel_pos(self, pos):
        return [pos[0] + self.CIRCLE_RADIUS_DEFAULT, pos[1] + self.CIRCLE_RADIUS_DEFAULT]


    # Shift edge w.r.t circle centers to have space for edges of both directions
    def edge_calc_rel_pos(self, p1, p2):
        vl = p2 - p1
        vl /= np.linalg.norm(vl)
        vr = np.array([ vl[1], -vl[0] ])  # Rotate vector 90 degrees
        p1sh = p1 + self.CIRCLE_RADIUS_DEFAULT / 2 * vr
        p2sh = p2 + self.CIRCLE_RADIUS_DEFAULT / 2 * vr
        return p1sh, p2sh


    def edge_calc_alpha_from_fc(self, fc):
        if (fc is None) or fc <= 0:
            raise ValueError("Unexpected FC value", fc)
        return np.min([2*fc, 1])

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
