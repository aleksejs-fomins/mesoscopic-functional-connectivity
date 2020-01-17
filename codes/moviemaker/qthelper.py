from PyQt5 import QtGui, QtCore, QtWidgets


def qt_check_state(isChecked):
    return QtCore.Qt.Checked if isChecked else QtCore.Qt.Unchecked

# Check if value already is a table item
# If it is not, convert it to one by converting it to a string
def qtable_parse_widget(val):
    if isinstance(val, QtWidgets.QTableWidgetItem):
        return val
    else:
        return QtWidgets.QTableWidgetItem(str(val))


# Constructs a checkbox that works as a cell of a qtable
def qtable_make_checkbox(checked = False):
    checkBox = QtWidgets.QTableWidgetItem()
    checkBox.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
    checkBox.setCheckState(qt_check_state(checked))
    return checkBox

# Fill in combo box
def qcombo_fill(qcombo, values):
    qcombo.clear()
    for val in values:
        qcombo.addItem(str(val))


# Add a row to an existing table
def qtable_addrow(qtable, iRow, values):
    qtable.insertRow(iRow)
    qtable_setrowval(qtable, iRow, values)


# Replace values of an existing row
# If raw is true, exact values are used, otherwise values are converted to strings
def qtable_setrowval(qtable, iRow, values):
    for iCol, val in enumerate(values):
        valWidget = qtable_parse_widget(val)
        qtable.setItem(iRow, iCol, valWidget)


# Replace values of an existing column
# If raw is true, exact values are used, otherwise values are converted to strings
def qtable_setcolval(qtable, iCol, values, raw=False):
    for iRow, val in enumerate(values):
        valWidget = qtable_parse_widget(val)
        qtable.setItem(iRow, iCol, valWidget)


# Flags determines which flags to be switched
# Val determines whether to switch up or down
def qtable_setcolflag(qtable, iCol, flags, val):
    for iRow in range(qtable.rowCount()):
        newFlags = (qtable.item(iRow, iCol).flags() & ~flags) | (flags & val)
        qtable.item(iRow, iCol).setFlags(newFlags)


def qtable_setrowcolor(qtable, iRow, color):
    for iCol in range(qtable.columnCount()):
        qtable.item(iRow, iCol).setBackground(color)


def qtable_setcolcolor(qtable, iCol, color):
    for iRow in range(qtable.rowCount()):
        qtable.item(iRow, iCol).setBackground(color)


def qtable_setcolchecked(qtable, iCol, checkedLst):
    for iRow, isChecked in enumerate(checkedLst):
        qtable.item(iRow, iCol).setCheckState(qt_check_state(isChecked))


def qtable_getcoltext(qtable, iCol):
    return [qtable.item(iRow, iCol).text() for iRow in range(qtable.rowCount())]


def qtable_getcolchecked(qtable, iCol):
    return [qtable.item(iRow, iCol).checkState() == QtCore.Qt.Checked for iRow in range(qtable.rowCount())]