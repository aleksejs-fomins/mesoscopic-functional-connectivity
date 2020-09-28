#include "moviemakerapp.h"
#include "ui_moviemakerapp.h"

MovieMakerApp::MovieMakerApp(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MovieMakerApp)
{
    ui->setupUi(this);
}

MovieMakerApp::~MovieMakerApp()
{
    delete ui;
}
