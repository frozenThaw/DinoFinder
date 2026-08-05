#pragma once

#include "core/app_types.h"

#include <QMainWindow>

class QLabel;
class QLineEdit;
class QListView;
class QCloseEvent;
class QDragEnterEvent;
class QDropEvent;
class QModelIndex;
class QProgressBar;
class QPushButton;
class QSlider;
class QSpinBox;
class QDoubleSpinBox;
class QThread;
class QTimer;
class ResultModel;

namespace sf {
class SimilarityService;
}

class MainWindow final : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override;

protected:
    void dragEnterEvent(QDragEnterEvent* event) override;
    void dropEvent(QDropEvent* event) override;
    void closeEvent(QCloseEvent* event) override;

private:
    void buildUi();
    void connectUi();
    void restoreSettings();
    void saveSettings() const;
    void chooseDirectory();
    void chooseImage();
    void setQueryImage(const QString& path);
    void startSearch();
    void setBusy(bool busy, bool building);
    void openResult(const QModelIndex& index);
    void showResultMenu(const QPoint& position);

    QThread* workerThread_ = nullptr;
    sf::SimilarityService* service_ = nullptr;
    ResultModel* resultModel_ = nullptr;

    QLineEdit* directoryEdit_ = nullptr;
    QPushButton* chooseDirectoryButton_ = nullptr;
    QPushButton* buildButton_ = nullptr;
    QPushButton* cancelButton_ = nullptr;
    QLabel* providerLabel_ = nullptr;
    QLabel* progressLabel_ = nullptr;
    QProgressBar* progressBar_ = nullptr;
    QLabel* queryPreview_ = nullptr;
    QPushButton* chooseImageButton_ = nullptr;
    QPushButton* searchButton_ = nullptr;
    QSlider* similaritySlider_ = nullptr;
    QDoubleSpinBox* similaritySpin_ = nullptr;
    QSlider* topSlider_ = nullptr;
    QSpinBox* topSpin_ = nullptr;
    QListView* resultsView_ = nullptr;
    QTimer* searchDebounce_ = nullptr;

    QString queryPath_;
    bool catalogBuilt_ = false;
    bool catalogReadOnly_ = false;
};
