#pragma once

#include "core/app_types.h"
#include "core/catalog_store.h"
#include "core/dino_engine.h"
#include "core/faiss_index.h"
#include "core/image_preprocessor.h"

#include <QObject>
#include <QString>

#include <atomic>
#include <memory>

namespace sf {

class SimilarityService final : public QObject {
    Q_OBJECT

public:
    explicit SimilarityService(QObject* parent = nullptr);
    ~SimilarityService() override;

    void requestCancel();

public Q_SLOTS:
    void initialize(const QString& applicationDirectory);
    void openDirectory(const QString& sourceDirectory);
    void buildCatalog();
    void search(const QString& queryPath, sf::SearchOptions options);

Q_SIGNALS:
    void runtimeReady(const QString& providerName);
    void catalogReady(const sf::CatalogSummary& summary);
    void buildProgress(const sf::BuildProgress& progress);
    void buildCompleted(const sf::CatalogSummary& summary, bool cancelled);
    void searchCompleted(
        const QString& queryPath,
        const QVector<sf::SearchResult>& results,
        qint64 elapsedMilliseconds);
    void failed(const QString& title, const QString& message);

private:
    void ensureRuntime();
    void loadIndexAndReplay();
    void saveCheckpoint();
    bool compactIfNeeded();
    CatalogSummary summary() const;

    QString applicationDirectory_;
    QString modelPath_;
    QString manifestPath_;
    QString modelHash_;
    ImagePreprocessor preprocessor_;
    DinoEngine engine_;
    CatalogStore store_;
    std::unique_ptr<FaissIndex> index_;
    std::atomic_bool cancelRequested_{false};
    bool runtimeInitialized_ = false;
    bool busy_ = false;
    std::int64_t checkpointSequence_ = 0;
};

}  // namespace sf
