#include "app/main_window.h"

#include "app/result_delegate.h"
#include "app/result_model.h"
#include "core/similarity_service.h"

#include <QApplication>
#include <QClipboard>
#include <QCloseEvent>
#include <QDesktopServices>
#include <QDir>
#include <QDoubleSpinBox>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QFileDialog>
#include <QFileInfo>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QImage>
#include <QImageReader>
#include <QLabel>
#include <QLineEdit>
#include <QListView>
#include <QMenu>
#include <QMessageBox>
#include <QMimeData>
#include <QProcess>
#include <QProgressBar>
#include <QPushButton>
#include <QSettings>
#include <QSet>
#include <QSizePolicy>
#include <QSlider>
#include <QSpinBox>
#include <QStatusBar>
#include <QTemporaryDir>
#include <QThread>
#include <QTimer>
#include <QUrl>
#include <QVBoxLayout>

#include <algorithm>

namespace {

bool isSupportedImage(const QString& path) {
    static const QSet<QString> extensions = {
        QStringLiteral("png"), QStringLiteral("jpg"), QStringLiteral("jpeg"),
        QStringLiteral("bmp"), QStringLiteral("webp"), QStringLiteral("tif"),
        QStringLiteral("tiff")};
    return QFileInfo(path).isFile() && extensions.contains(QFileInfo(path).suffix().toLower());
}

QString formatEta(const qint64 seconds) {
    if (seconds < 0) {
        return QStringLiteral("--:--");
    }
    return QStringLiteral("%1:%2")
        .arg(seconds / 60, 2, 10, QLatin1Char('0'))
        .arg(seconds % 60, 2, 10, QLatin1Char('0'));
}

}  // namespace

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    buildUi();
    resultModel_ = new ResultModel(this);
    resultsView_->setModel(resultModel_);
    resultsView_->setItemDelegate(new ResultDelegate(resultsView_));

    workerThread_ = new QThread(this);
    service_ = new sf::SimilarityService();
    service_->moveToThread(workerThread_);
    connect(workerThread_, &QThread::finished, service_, &QObject::deleteLater);
    connectUi();
    workerThread_->start();
    QMetaObject::invokeMethod(service_, [this]() {
        service_->initialize(QCoreApplication::applicationDirPath());
    });
    restoreSettings();
}

MainWindow::~MainWindow() {
    service_->requestCancel();
    workerThread_->quit();
    workerThread_->wait();
}

void MainWindow::buildUi() {
    setWindowTitle(QStringLiteral("SimilarFinder"));
    setMinimumSize(920, 680);
    resize(1240, 860);
    setAcceptDrops(true);

    auto* central = new QWidget(this);
    auto* root = new QVBoxLayout(central);
    root->setContentsMargins(18, 16, 18, 16);
    root->setSpacing(12);

    auto* directoryRow = new QHBoxLayout();
    directoryEdit_ = new QLineEdit();
    directoryEdit_->setReadOnly(true);
    directoryEdit_->setPlaceholderText(QStringLiteral("Choose an image directory to build a catalog"));
    chooseDirectoryButton_ = new QPushButton(QStringLiteral("Choose Directory"));
    buildButton_ = new QPushButton(QStringLiteral("Build"));
    buildButton_->setEnabled(false);
    cancelButton_ = new QPushButton(QStringLiteral("Cancel"));
    cancelButton_->setEnabled(false);
    directoryRow->addWidget(directoryEdit_, 1);
    directoryRow->addWidget(chooseDirectoryButton_);
    directoryRow->addWidget(buildButton_);
    directoryRow->addWidget(cancelButton_);
    root->addLayout(directoryRow);

    auto* progressRow = new QHBoxLayout();
    progressLabel_ = new QLabel(QStringLiteral("Ready"));
    progressBar_ = new QProgressBar();
    progressBar_->setTextVisible(false);
    progressBar_->setRange(0, 100);
    progressBar_->setValue(0);
    providerLabel_ = new QLabel(QStringLiteral("Runtime: loading"));
    providerLabel_->setObjectName(QStringLiteral("providerBadge"));
    progressRow->addWidget(progressLabel_);
    progressRow->addWidget(progressBar_, 1);
    progressRow->addWidget(providerLabel_);
    root->addLayout(progressRow);

    auto* queryBox = new QGroupBox(QStringLiteral("Query Image"));
    auto* queryLayout = new QHBoxLayout(queryBox);
    queryPreview_ = new QLabel(QStringLiteral("Drop an image here"));
    queryPreview_->setAlignment(Qt::AlignCenter);
    queryPreview_->setMinimumSize(300, 180);
    queryPreview_->setFixedHeight(180);
    queryPreview_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    queryPreview_->setObjectName(QStringLiteral("queryPreview"));
    queryLayout->addWidget(queryPreview_, 1);

    auto* querySidePanel = new QWidget();
    querySidePanel->setFixedSize(480, 180);
    auto* querySideLayout = new QVBoxLayout(querySidePanel);
    querySideLayout->setContentsMargins(0, 0, 0, 0);
    querySideLayout->setSpacing(8);

    auto* queryButtons = new QVBoxLayout();
    queryButtons->setSpacing(8);
    chooseImageButton_ = new QPushButton(QStringLiteral("Choose Image"));
    chooseImageButton_->setFixedWidth(140);
    pasteImageButton_ = new QPushButton(QStringLiteral("Paste Image"));
    pasteImageButton_->setFixedWidth(140);
    queryButtons->addWidget(chooseImageButton_, 0, Qt::AlignLeft);
    queryButtons->addWidget(pasteImageButton_, 0, Qt::AlignLeft);
    querySideLayout->addLayout(queryButtons);

    auto* controlsPanel = new QWidget();
    controlsPanel->setObjectName(QStringLiteral("queryControls"));
    controlsPanel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    auto* controls = new QGridLayout(controlsPanel);
    controls->setContentsMargins(14, 12, 14, 12);
    controls->setHorizontalSpacing(10);
    controls->setVerticalSpacing(14);
    controls->setColumnStretch(1, 1);
    similaritySlider_ = new QSlider(Qt::Horizontal);
    similaritySlider_->setRange(50, 100);
    similaritySlider_->setValue(80);
    similaritySpin_ = new QDoubleSpinBox();
    similaritySpin_->setRange(0.50, 1.0);
    similaritySpin_->setSingleStep(0.01);
    similaritySpin_->setDecimals(2);
    similaritySpin_->setValue(0.80);
    similaritySpin_->setFixedWidth(82);
    topSlider_ = new QSlider(Qt::Horizontal);
    topSlider_->setRange(1, 1000);
    topSlider_->setValue(200);
    topSpin_ = new QSpinBox();
    topSpin_->setRange(1, 1000);
    topSpin_->setValue(200);
    topSpin_->setFixedWidth(82);
    controls->addWidget(new QLabel(QStringLiteral("Similar Value")), 0, 0);
    controls->addWidget(similaritySlider_, 0, 1);
    controls->addWidget(similaritySpin_, 0, 2);
    controls->addWidget(new QLabel(QStringLiteral("Top Num")), 1, 0);
    controls->addWidget(topSlider_, 1, 1);
    controls->addWidget(topSpin_, 1, 2);
    querySideLayout->addWidget(controlsPanel);
    queryLayout->addWidget(querySidePanel);
    root->addWidget(queryBox);

    auto* resultHeader = new QHBoxLayout();
    auto* resultTitle = new QLabel(QStringLiteral("Search Results"));
    QFont titleFont = resultTitle->font();
    titleFont.setPointSize(12);
    titleFont.setBold(true);
    resultTitle->setFont(titleFont);
    resultHeader->addWidget(resultTitle);
    resultHeader->addStretch();
    root->addLayout(resultHeader);

    resultsView_ = new QListView();
    resultsView_->setViewMode(QListView::IconMode);
    resultsView_->setResizeMode(QListView::Adjust);
    resultsView_->setMovement(QListView::Static);
    resultsView_->setWrapping(true);
    resultsView_->setUniformItemSizes(true);
    resultsView_->setGridSize(QSize(222, 256));
    resultsView_->setSpacing(2);
    resultsView_->setSelectionMode(QAbstractItemView::SingleSelection);
    resultsView_->setContextMenuPolicy(Qt::CustomContextMenu);
    root->addWidget(resultsView_, 1);

    setCentralWidget(central);
    statusBar()->showMessage(QStringLiteral("Choose a directory to begin."));
    setStyleSheet(QStringLiteral(R"(
        QMainWindow, QWidget { background: #ffffff; color: #171a1d; }
        QLineEdit, QSpinBox, QDoubleSpinBox { min-height: 30px; border: 1px solid #cfd5d9; padding: 0 8px; }
        QPushButton { min-height: 32px; padding: 0 14px; border: 1px solid #bfc6ca; background: #f5f7f7; border-radius: 4px; }
        QPushButton:hover { background: #e9eeee; }
        QPushButton:disabled { color: #9da4a9; background: #f7f8f8; }
        QGroupBox { border: 1px solid #dfe3e6; margin-top: 10px; padding-top: 10px; }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; font-weight: 600; }
        QLabel#providerBadge { color: #087f5b; font-weight: 600; padding: 4px 8px; border: 1px solid #9dd5c4; background: #eef8f4; }
        QLabel#queryPreview { border: 1px dashed #adb5ba; background: #f5f7f7; color: #687077; }
        QWidget#queryControls { border: 1px solid #dfe3e6; background: #fafbfb; }
        QProgressBar { height: 8px; border: 0; background: #e8ebed; }
        QProgressBar::chunk { background: #087f5b; }
        QListView { border: 0; border-top: 1px solid #dfe3e6; outline: 0; }
    )"));
}

void MainWindow::connectUi() {
    searchDebounce_ = new QTimer(this);
    searchDebounce_->setSingleShot(true);
    searchDebounce_->setInterval(150);
    connect(searchDebounce_, &QTimer::timeout, this, &MainWindow::startSearch);
    connect(chooseDirectoryButton_, &QPushButton::clicked, this, &MainWindow::chooseDirectory);
    connect(chooseImageButton_, &QPushButton::clicked, this, &MainWindow::chooseImage);
    connect(pasteImageButton_, &QPushButton::clicked, this, &MainWindow::pasteImage);
    connect(buildButton_, &QPushButton::clicked, this, [this]() {
        setBusy(true, true);
        QMetaObject::invokeMethod(service_, &sf::SimilarityService::buildCatalog, Qt::QueuedConnection);
    });
    connect(cancelButton_, &QPushButton::clicked, this, [this]() {
        progressLabel_->setText(QStringLiteral("Cancelling after the current batch..."));
        service_->requestCancel();
    });
    connect(similaritySlider_, &QSlider::valueChanged, this, [this](const int value) {
        similaritySpin_->setValue(value / 100.0);
        if (!queryPath_.isEmpty() && catalogBuilt_) {
            searchDebounce_->start();
        }
    });
    connect(similaritySpin_, &QDoubleSpinBox::valueChanged, this, [this](const double value) {
        similaritySlider_->setValue(qRound(value * 100.0));
    });
    connect(topSlider_, &QSlider::valueChanged, this, [this](const int value) {
        topSpin_->setValue(value);
        if (!queryPath_.isEmpty() && catalogBuilt_) {
            searchDebounce_->start();
        }
    });
    connect(topSpin_, &QSpinBox::valueChanged, topSlider_, &QSlider::setValue);
    connect(resultsView_, &QListView::doubleClicked, this, &MainWindow::openResult);
    connect(resultsView_, &QListView::customContextMenuRequested, this, &MainWindow::showResultMenu);

    connect(service_, &sf::SimilarityService::runtimeReady, this, [this](const QString& provider) {
        providerLabel_->setText(QStringLiteral("Runtime: %1").arg(provider));
    });
    connect(service_, &sf::SimilarityService::catalogReady, this, [this](const sf::CatalogSummary& summary) {
        directoryEdit_->setText(summary.sourceDirectory);
        catalogBuilt_ = summary.activeVectorCount > 0;
        catalogReadOnly_ = summary.readOnly;
        buildButton_->setText(summary.imageCount > 0 ? QStringLiteral("Sync") : QStringLiteral("Build"));
        buildButton_->setEnabled(!summary.readOnly);
        statusBar()->showMessage(
            QStringLiteral("%1 images, %2 vectors%3")
                .arg(summary.imageCount)
                .arg(summary.activeVectorCount)
                .arg(summary.readOnly ? QStringLiteral(" · read-only") : QString()));
        if (catalogBuilt_ && !queryPath_.isEmpty()) {
            startSearch();
        }
    });
    connect(service_, &sf::SimilarityService::buildProgress, this, [this](const sf::BuildProgress& progress) {
        const int percent = progress.total > 0
            ? static_cast<int>(100 * progress.processed / progress.total)
            : 0;
        progressBar_->setValue(std::clamp(percent, 0, 100));
        progressLabel_->setText(
            QStringLiteral("%1 · %2/%3 · %4 img/s · ETA %5")
                .arg(progress.stage)
                .arg(progress.processed)
                .arg(progress.total)
                .arg(progress.itemsPerSecond, 0, 'f', 1)
                .arg(formatEta(progress.etaSeconds)));
    });
    connect(service_, &sf::SimilarityService::buildCompleted, this,
        [this](const sf::CatalogSummary& summary, const bool cancelled) {
            setBusy(false, false);
            catalogBuilt_ = summary.activeVectorCount > 0;
            buildButton_->setText(summary.imageCount > 0 ? QStringLiteral("Sync") : QStringLiteral("Build"));
            progressLabel_->setText(cancelled ? QStringLiteral("Build paused") : QStringLiteral("Build complete"));
            statusBar()->showMessage(
                QStringLiteral("%1 images, %2 vectors").arg(summary.imageCount).arg(summary.activeVectorCount));
            if (!cancelled && catalogBuilt_ && !queryPath_.isEmpty()) {
                startSearch();
            }
        });
    connect(service_, &sf::SimilarityService::searchCompleted, this,
        [this](const QString&, const QVector<sf::SearchResult>& results, const qint64 elapsed) {
            setBusy(false, false);
            resultModel_->setResults(results);
            statusBar()->showMessage(
                QStringLiteral("%1 result%2 · %3 ms")
                    .arg(results.size())
                    .arg(results.size() == 1 ? QString() : QStringLiteral("s"))
                    .arg(elapsed));
        });
    connect(service_, &sf::SimilarityService::failed, this, [this](const QString& title, const QString& message) {
        setBusy(false, false);
        QMessageBox::critical(this, title, message);
    });
}

void MainWindow::restoreSettings() {
    QSettings settings;
    restoreGeometry(settings.value(QStringLiteral("window/geometry")).toByteArray());
    similaritySpin_->setValue(settings.value(QStringLiteral("search/similarity"), 0.80).toDouble());
    topSpin_->setValue(settings.value(QStringLiteral("search/topNum"), 200).toInt());
    const QString directory = settings.value(QStringLiteral("catalog/lastDirectory")).toString();
    if (QFileInfo(directory).isDir()) {
        directoryEdit_->setText(directory);
        QMetaObject::invokeMethod(service_, [this, directory]() {
            service_->openDirectory(directory);
        }, Qt::QueuedConnection);
    }
}

void MainWindow::saveSettings() const {
    QSettings settings;
    settings.setValue(QStringLiteral("window/geometry"), saveGeometry());
    settings.setValue(QStringLiteral("search/similarity"), similaritySpin_->value());
    settings.setValue(QStringLiteral("search/topNum"), topSpin_->value());
    settings.setValue(QStringLiteral("catalog/lastDirectory"), directoryEdit_->text());
}

void MainWindow::chooseDirectory() {
    const QString directory = QFileDialog::getExistingDirectory(
        this,
        QStringLiteral("Choose image directory"),
        directoryEdit_->text());
    if (directory.isEmpty()) {
        return;
    }
    directoryEdit_->setText(QDir::cleanPath(directory));
    catalogBuilt_ = false;
    buildButton_->setEnabled(false);
    QMetaObject::invokeMethod(service_, [this, directory]() {
        service_->openDirectory(directory);
    }, Qt::QueuedConnection);
}

void MainWindow::chooseImage() {
    const QString path = QFileDialog::getOpenFileName(
        this,
        QStringLiteral("Choose query image"),
        QString(),
        QStringLiteral("Images (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff)"));
    if (!path.isEmpty()) {
        setQueryImage(path);
    }
}

void MainWindow::pasteImage() {
    const QClipboard* clipboard = QApplication::clipboard();
    const QImage image = clipboard->image();
    if (!image.isNull()) {
        if (!clipboardDirectory_) {
            clipboardDirectory_ = std::make_unique<QTemporaryDir>(
                QDir(QDir::tempPath()).filePath(QStringLiteral("SimilarFinder-clipboard-XXXXXX")));
        }
        if (!clipboardDirectory_->isValid()) {
            QMessageBox::warning(
                this,
                QStringLiteral("Paste failed"),
                QStringLiteral("Cannot create a temporary directory for the clipboard image."));
            clipboardDirectory_.reset();
            return;
        }
        const QString path = QDir(clipboardDirectory_->path()).filePath(
            QStringLiteral("clipboard-query-%1.png").arg(++clipboardImageSequence_));
        if (!image.save(path, "PNG")) {
            QMessageBox::warning(
                this,
                QStringLiteral("Paste failed"),
                QStringLiteral("Cannot save the clipboard image."));
            return;
        }
        setQueryImage(path);
        return;
    }

    const QMimeData* mimeData = clipboard->mimeData();
    if (mimeData != nullptr && mimeData->hasUrls()) {
        for (const QUrl& url : mimeData->urls()) {
            if (url.isLocalFile() && isSupportedImage(url.toLocalFile())) {
                setQueryImage(url.toLocalFile());
                return;
            }
        }
    }
    QMessageBox::information(
        this,
        QStringLiteral("No image to paste"),
        QStringLiteral("The clipboard does not contain a supported image."));
}

void MainWindow::setQueryImage(const QString& path) {
    if (!isSupportedImage(path)) {
        QMessageBox::warning(this, QStringLiteral("Unsupported image"), QStringLiteral("Choose a supported image file."));
        return;
    }
    queryPath_ = QFileInfo(path).absoluteFilePath();
    QImageReader reader(queryPath_);
    reader.setAutoTransform(true);
    reader.setScaledSize(reader.size().scaled(queryPreview_->size() - QSize(12, 12), Qt::KeepAspectRatio));
    const QImage preview = reader.read();
    queryPreview_->setPixmap(QPixmap::fromImage(preview));
    queryPreview_->setToolTip(queryPath_);
    if (catalogBuilt_) {
        startSearch();
    }
}

void MainWindow::startSearch() {
    if (!catalogBuilt_ || queryPath_.isEmpty()) {
        return;
    }
    sf::SearchOptions options;
    options.minSimilarity = static_cast<float>(similaritySpin_->value());
    options.topNum = topSpin_->value();
    setBusy(true, false);
    const QString path = queryPath_;
    QMetaObject::invokeMethod(service_, [this, path, options]() {
        service_->search(path, options);
    }, Qt::QueuedConnection);
}

void MainWindow::setBusy(const bool busy, const bool building) {
    chooseDirectoryButton_->setEnabled(!busy);
    buildButton_->setEnabled(!busy && !catalogReadOnly_ && !directoryEdit_->text().isEmpty());
    cancelButton_->setEnabled(busy && building);
    chooseImageButton_->setEnabled(!building);
    pasteImageButton_->setEnabled(!building);
    similaritySlider_->setEnabled(!busy);
    similaritySpin_->setEnabled(!busy);
    topSlider_->setEnabled(!busy);
    topSpin_->setEnabled(!busy);
}

void MainWindow::openResult(const QModelIndex& index) {
    const QString path = index.data(ResultModel::AbsolutePathRole).toString();
    if (!path.isEmpty()) {
        QDesktopServices::openUrl(QUrl::fromLocalFile(path));
    }
}

void MainWindow::showResultMenu(const QPoint& position) {
    const QModelIndex index = resultsView_->indexAt(position);
    if (!index.isValid()) {
        return;
    }
    const QString path = index.data(ResultModel::AbsolutePathRole).toString();
    QMenu menu(this);
    QAction* show = menu.addAction(QStringLiteral("Show in Explorer"));
    QAction* copy = menu.addAction(QStringLiteral("Copy Path"));
    QAction* chosen = menu.exec(resultsView_->viewport()->mapToGlobal(position));
    if (chosen == show) {
        QProcess::startDetached(
            QStringLiteral("explorer.exe"),
            {QStringLiteral("/select,%1").arg(QDir::toNativeSeparators(path))});
    } else if (chosen == copy) {
        QApplication::clipboard()->setText(path);
    }
}

void MainWindow::dragEnterEvent(QDragEnterEvent* event) {
    if (event->mimeData()->hasUrls()) {
        for (const QUrl& url : event->mimeData()->urls()) {
            if (url.isLocalFile() && isSupportedImage(url.toLocalFile())) {
                event->acceptProposedAction();
                return;
            }
        }
    }
}

void MainWindow::dropEvent(QDropEvent* event) {
    for (const QUrl& url : event->mimeData()->urls()) {
        if (url.isLocalFile() && isSupportedImage(url.toLocalFile())) {
            setQueryImage(url.toLocalFile());
            event->acceptProposedAction();
            return;
        }
    }
}

void MainWindow::closeEvent(QCloseEvent* event) {
    saveSettings();
    service_->requestCancel();
    QMainWindow::closeEvent(event);
}
