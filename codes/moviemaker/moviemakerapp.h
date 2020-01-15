#ifndef MOVIEMAKERAPP_H
#define MOVIEMAKERAPP_H

#include <QMainWindow>

namespace Ui {
class MovieMakerApp;
}

class MovieMakerApp : public QMainWindow
{
    Q_OBJECT

public:
    explicit MovieMakerApp(QWidget *parent = 0);
    ~MovieMakerApp();

private:
    Ui::MovieMakerApp *ui;
};

#endif // MOVIEMAKERAPP_H
