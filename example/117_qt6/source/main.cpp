#include <QApplication>
#include <QFile>
#include <QTextStream>
#include <QtCharts/QChartView>
#include <QtCharts/QLineSeries>
#include <QtWidgets/QMainWindow>

// QT_CHARTS_USE_NAMESPACE
// using namespace QtCharts;

QT_USE_NAMESPACE

int main(int argc, char *argv[]) {
    // Create a new Qt application
    QApplication app(argc, argv);

    // Create a new Qt window
    QMainWindow window;
    window.resize(800, 600);
    window.setWindowTitle("CO2 Concentration vs. Time");

    // Create a new Qt chart
    QChart *chart = new QChart();
    chart->setTitle("CO2 Concentration vs. Time");

    // Create a new Qt line series to hold the data
    QLineSeries *series = new QLineSeries();

    // Open the data file and read in the time and CO2 concentration values
    QFile file("data.txt");
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        return 1;
    }

    QTextStream in(&file);
    while (!in.atEnd()) {
        QString line = in.readLine();
        QStringList values = line.split(" ");

        double time = values.at(0).toDouble();
        double concentration = values.at(1).toDouble();

        series->append(time, concentration);
    }

    file.close();

    // Add the line series to the chart
    chart->addSeries(series);

    // Set up the X and Y axes
    QValueAxis *axisX = new QValueAxis;
    axisX->setTitleText("Time");
    chart->addAxis(axisX, Qt::AlignBottom);
    series->attachAxis(axisX);

    QValueAxis *axisY = new QValueAxis;
    axisY->setTitleText("CO2 Concentration");
    chart->addAxis(axisY, Qt::AlignLeft);
    series->attachAxis(axisY);

    // Create a new Qt chart view to display the chart in the window
    QChartView *chartView = new QChartView(chart);
    chartView->setRenderHint(QPainter::Antialiasing);
    window.setCentralWidget(chartView);

    // Show the window and start the Qt event loop
    window.show();
    return app.exec();
}
