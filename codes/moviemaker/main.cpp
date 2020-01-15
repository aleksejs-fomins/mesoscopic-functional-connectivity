#include "moviemakerapp.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MovieMakerApp w;
    w.show();

    return a.exec();
}
