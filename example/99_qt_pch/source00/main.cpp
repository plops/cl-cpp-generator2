#include <QDoubleValidator>
#include <QGroupBox>
#include <QMainWindow>
void convertTemperature(QLineEdit *outputLineEdit, QComboBox *outputComboBox,
                        const QString &inputUnit, double inputTemp) {
  auto outputTemp = double((0.));
  if (("Celsius") == (inputUnit)) {
    outputTemp = inputTemp;
  } else {
    if (("Fahrenheit") == (inputUnit)) {
      outputTemp = ((((5) * (((inputTemp) - (32))))) / (9));
    } else {
      if (("Kelvin") == (inputUnit)) {
        outputTemp = ((inputTemp) - ((273.150000000000000000000000000)));
      }
    }
  }
  auto outputUnit = outputComboBox->currentText();
  if (("Celsius") == (outputUnit)) {
    outputTemp = outputTemp;
  } else {
    if (("Fahrenheit") == (outputUnit)) {
      outputTemp = ((32) + (((((9) * (outputTemp))) / (5))));
    } else {
      if (("Kelvin") == (inputUnit)) {
        outputTemp = ((outputTemp) + ((273.150000000000000000000000000)));
      }
    }
  }
  outputLineEdit->setText(QString::number(outputTemp));
}
int main(int argc, char **argv) {
  (void)argv;
  spdlog::info("start  argc='{}'", argc);
  auto app = QApplication(argc, argv);
  auto window = QMainWindow();
  window.setWindowTitle("Temperature converter");
  auto *mainLayout = new QVBoxLayout;
  auto *inputGroupBox = new QGroupBox("Input");
  auto *inputLayout = new QHBoxLayout;
  auto *inputLineEdit = new QLineEdit;
  auto *inputValidator = new QDoubleValidator;
  inputLineEdit->setValidator(inputValidator);
  auto *inputComboBox = new QComboBox;
  inputComboBox->addItem("Celsius");
  inputComboBox->addItem("Fahrenheit");
  inputComboBox->addItem("Kelvin");
  inputLayout->addWidget(inputLineEdit);
  inputLayout->addWidget(inputComboBox);
  inputGroupBox->setLayout(inputLayout);
  mainLayout->addWidget(inputGroupBox);
  auto *outputGroupBox = new QGroupBox("Output");
  auto *outputLayout = new QHBoxLayout;
  auto *outputLineEdit = new QLineEdit;
  outputLineEdit->setReadOnly(true);
  auto *outputValidator = new QDoubleValidator;
  outputLineEdit->setValidator(outputValidator);
  auto *outputComboBox = new QComboBox;
  outputComboBox->addItem("Celsius");
  outputComboBox->addItem("Fahrenheit");
  outputComboBox->addItem("Kelvin");
  outputLayout->addWidget(outputLineEdit);
  outputLayout->addWidget(outputComboBox);
  outputGroupBox->setLayout(outputLayout);
  mainLayout->addWidget(outputGroupBox);
  auto *convertButton = new QPushButton("Convert");
  QObject::connect(convertButton, &QPushButton::clicked, [=]() {
    auto inputTemp = inputLineEdit->text().toDouble();
    auto inputUnit = inputComboBox->currentText();
    convertTemperature(outputLineEdit, outputComboBox, inputUnit, intputTemp);
  });
  mainLayout->addWidget(convertButton);
  window.setLayout(mainLayout);
  window.show();
  return app.exec();
}