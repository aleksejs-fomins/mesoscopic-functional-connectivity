from sys import executable, argv
from subprocess import check_output
from PyQt5.QtWidgets import QFileDialog, QApplication

def gui_fsave(title = "Select a file...", directory='./', filter="All files (*)"):
    return check_output([executable, __file__, "save", title, directory, filter]).strip().decode('UTF-8')

def gui_fname(title = "Select a file...", directory='./', filter="All files (*)"):
    return check_output([executable, __file__, "file", title, directory, filter]).strip().decode('UTF-8')

def gui_fnames(title = "Select a file...", directory='./', filter="All files (*)"):
    output = check_output([executable, __file__, "files", title, directory, filter]).strip().decode('UTF-8')
    return output[1:-1].replace("'", "").replace(' ', '').split(',')

def gui_fpath(title="Select a path...", directory='./'):
    return check_output([executable, __file__, "folder", title, directory]).strip().decode('UTF-8')

if __name__ == "__main__":
    param = argv[1:]
    app = QApplication(param)

    if param[0] == "save":
        title, directory, filter = param[1:]
        print(QFileDialog.getSaveFileName(None, title, directory, filter=filter)[0])
    if param[0] == "file":
        title, directory, filter = param[1:]
        print(QFileDialog.getOpenFileName(None, title, directory, filter=filter)[0])
    elif param[0] == "files":
        title, directory, filter = param[1:]
        print(QFileDialog.getOpenFileNames(None, title, directory, filter=filter)[0])
    elif param[0] == "folder":
        title, directory = param[1:]
        print(QFileDialog.getExistingDirectory(None, title, directory))
    else:
        raise ValueError("Unexpected operation", param[0])