#include "core/similarity_service.h"

#include "core/feature_utils.h"

#include <QDateTime>
#include <QDir>
#include <QDirIterator>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSet>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sf {

namespace {

struct SourceFile {
    QString absolutePath;
    QString relativePath;
    std::int64_t size = 0;
    std::int64_t modifiedMilliseconds = 0;
    FileDecision decision;
};

struct BatchImage {
    SourceFile file;
    PreparedImage prepared;
    QString error;
    std::size_t embeddingOffset = 0;
};

const QStringList kImageFilters = {
    QStringLiteral("*.png"),
    QStringLiteral("*.jpg"),
    QStringLiteral("*.jpeg"),
    QStringLiteral("*.bmp"),
    QStringLiteral("*.webp"),
    QStringLiteral("*.tif"),
    QStringLiteral("*.tiff"),
};

QString readModelHash(const QString& manifestPath) {
    QFile file(manifestPath);
    if (!file.open(QIODevice::ReadOnly)) {
        throw std::runtime_error(("Cannot open model manifest: " + manifestPath).toStdString());
    }
    const QString hash = QJsonDocument::fromJson(file.readAll())
                             .object()
                             .value(QStringLiteral("sha256"))
                             .toString()
                             .toLower();
    if (hash.size() != 64) {
        throw std::runtime_error("Model manifest SHA-256 is invalid");
    }
    return hash;
}

}  // namespace

SimilarityService::SimilarityService(QObject* parent) : QObject(parent) {}
SimilarityService::~SimilarityService() = default;

void SimilarityService::requestCancel() {
    cancelRequested_.store(true, std::memory_order_relaxed);
}

void SimilarityService::initialize(const QString& applicationDirectory) {
    applicationDirectory_ = QDir::cleanPath(applicationDirectory);
    modelPath_ = QDir(applicationDirectory_).filePath(QStringLiteral("models/dinov2_vits14.onnx"));
    manifestPath_ = QDir(applicationDirectory_).filePath(
        QStringLiteral("models/dinov2_vits14_manifest.json"));
    try {
        ensureRuntime();
    } catch (const std::exception& error) {
        Q_EMIT failed(QStringLiteral("Runtime initialization failed"), QString::fromUtf8(error.what()));
    }
}

void SimilarityService::ensureRuntime() {
    if (runtimeInitialized_) {
        return;
    }
    modelHash_ = readModelHash(manifestPath_);
    QString error;
    if (!engine_.initialize(modelPath_, manifestPath_, true, &error)) {
        throw std::runtime_error(error.toStdString());
    }
    runtimeInitialized_ = true;
    Q_EMIT runtimeReady(engine_.providerName());
}

void SimilarityService::openDirectory(const QString& sourceDirectory) {
    if (busy_) {
        Q_EMIT failed(QStringLiteral("Catalog busy"), QStringLiteral("Wait for the current operation to finish."));
        return;
    }
    try {
        ensureRuntime();
        store_.open(sourceDirectory, modelHash_);
        loadIndexAndReplay();
        Q_EMIT catalogReady(summary());
    } catch (const std::exception& error) {
        Q_EMIT failed(QStringLiteral("Cannot open catalog"), QString::fromUtf8(error.what()));
    }
}

void SimilarityService::loadIndexAndReplay() {
    index_ = std::make_unique<FaissIndex>();
    const IndexCheckpoint current = store_.checkpoint();
    if (!current.path.isEmpty()) {
        index_->load(current.path);
        checkpointSequence_ = current.maxVectorSequence;
    } else {
        index_->create();
        checkpointSequence_ = 0;
    }

    std::int64_t cursor = checkpointSequence_;
    while (true) {
        const QVector<StoredFeature> rows = store_.activeFeatureBatch(cursor, 4096);
        if (rows.isEmpty()) {
            break;
        }
        std::vector<std::int64_t> ids;
        std::vector<std::vector<float>> embeddings;
        ids.reserve(rows.size());
        embeddings.reserve(rows.size());
        for (const StoredFeature& row : rows) {
            cursor = std::max(cursor, row.sequence);
            ids.push_back(row.vectorId);
            embeddings.push_back(row.embedding);
        }
        index_->add(ids, embeddings);
    }
    const std::int64_t databaseSequence = store_.maxVectorSequence();
    if (!store_.isReadOnly() && databaseSequence > checkpointSequence_) {
        checkpointSequence_ = databaseSequence;
        saveCheckpoint();
    }
}

void SimilarityService::buildCatalog() {
    if (busy_ || !store_.isOpen() || store_.isReadOnly()) {
        Q_EMIT failed(
            QStringLiteral("Build unavailable"),
            store_.isReadOnly()
                ? QStringLiteral("This catalog is open in another application instance.")
                : QStringLiteral("Choose a writable source directory first."));
        return;
    }
    busy_ = true;
    cancelRequested_.store(false, std::memory_order_relaxed);
    QElapsedTimer timer;
    timer.start();
    try {
        loadIndexAndReplay();
        QVector<SourceFile> files;
        QDirIterator iterator(
            store_.sourceDirectory(),
            kImageFilters,
            QDir::Files | QDir::NoSymLinks,
            QDirIterator::Subdirectories);
        while (iterator.hasNext()) {
            if (cancelRequested_.load(std::memory_order_relaxed)) {
                busy_ = false;
                Q_EMIT buildCompleted(summary(), true);
                return;
            }
            iterator.next();
            const QFileInfo info = iterator.fileInfo();
            SourceFile file;
            file.absolutePath = info.absoluteFilePath();
            file.relativePath = QDir(store_.sourceDirectory()).relativeFilePath(file.absolutePath);
            file.size = info.size();
            file.modifiedMilliseconds = info.lastModified().toMSecsSinceEpoch();
            files.push_back(std::move(file));
        }

        const std::int64_t scanId = store_.beginScan();
        QVector<SourceFile> pending;
        pending.reserve(files.size());
        for (int index = 0; index < files.size(); ++index) {
            SourceFile& file = files[index];
            file.decision = store_.inspectFile(
                file.relativePath,
                file.size,
                file.modifiedMilliseconds,
                scanId);
            if (file.decision.requiresFeatures) {
                pending.push_back(file);
            }
            if ((index + 1) % 2000 == 0 || index + 1 == files.size()) {
                BuildProgress progress;
                progress.stage = QStringLiteral("Scanning");
                progress.processed = index + 1;
                progress.total = files.size();
                Q_EMIT buildProgress(progress);
            }
            if (cancelRequested_.load(std::memory_order_relaxed)) {
                busy_ = false;
                Q_EMIT buildCompleted(summary(), true);
                return;
            }
        }

        std::int64_t vectorsSinceCheckpoint = 0;
        QElapsedTimer checkpointTimer;
        checkpointTimer.start();
        constexpr int sourceBatchSize = 32;
        for (int start = 0; start < pending.size(); start += sourceBatchSize) {
            const int count = std::min(sourceBatchSize, static_cast<int>(pending.size()) - start);
            std::vector<BatchImage> batch;
            std::vector<ImageTensor> tensors;
            batch.reserve(count);
            for (int offset = 0; offset < count; ++offset) {
                BatchImage item;
                item.file = pending[start + offset];
                item.embeddingOffset = tensors.size();
                try {
                    item.prepared = preprocessor_.prepare(item.file.absolutePath);
                    for (ImageTensor& tensor : item.prepared.tensors) {
                        tensors.push_back(std::move(tensor));
                    }
                } catch (const std::exception& error) {
                    item.error = QString::fromUtf8(error.what());
                }
                batch.push_back(std::move(item));
            }

            std::vector<std::vector<float>> embeddings;
            if (!tensors.empty()) {
                embeddings = engine_.embed(tensors);
                Q_EMIT runtimeReady(engine_.providerName());
            }

            std::vector<std::int64_t> faissIds;
            std::vector<std::vector<float>> faissEmbeddings;
            store_.beginWriteBatch();
            try {
                for (const BatchImage& item : batch) {
                    if (!item.error.isEmpty()) {
                        store_.recordImageError(item.file.decision, item.error);
                        continue;
                    }
                    const std::size_t variantCount = item.prepared.tensors.size();
                    std::vector<BackgroundVariant> variants;
                    std::vector<std::vector<float>> imageEmbeddings;
                    variants.reserve(variantCount);
                    imageEmbeddings.reserve(variantCount);
                    for (std::size_t variant = 0; variant < variantCount; ++variant) {
                        variants.push_back(tensors[item.embeddingOffset + variant].variant);
                        imageEmbeddings.push_back(embeddings[item.embeddingOffset + variant]);
                    }
                    const auto ids = store_.commitImageFeatures(
                        item.file.decision,
                        item.file.size,
                        item.file.modifiedMilliseconds,
                        item.prepared.width,
                        item.prepared.height,
                        item.prepared.hasTransparency,
                        variants,
                        imageEmbeddings);
                    faissIds.insert(faissIds.end(), ids.begin(), ids.end());
                    faissEmbeddings.insert(
                        faissEmbeddings.end(),
                        imageEmbeddings.begin(),
                        imageEmbeddings.end());
                }
                store_.commitWriteBatch();
            } catch (...) {
                store_.rollbackWriteBatch();
                throw;
            }
            index_->add(faissIds, faissEmbeddings);
            vectorsSinceCheckpoint += static_cast<std::int64_t>(faissIds.size());

            const std::int64_t processed = start + count;
            const double seconds = std::max(0.001, timer.elapsed() / 1000.0);
            BuildProgress progress;
            progress.stage = QStringLiteral("Extracting features");
            progress.processed = processed;
            progress.total = pending.size();
            progress.itemsPerSecond = processed / seconds;
            progress.etaSeconds = progress.itemsPerSecond > 0.0
                ? static_cast<qint64>((pending.size() - processed) / progress.itemsPerSecond)
                : -1;
            Q_EMIT buildProgress(progress);

            if (vectorsSinceCheckpoint > 0
                && (vectorsSinceCheckpoint >= kCheckpointVectorCount
                    || checkpointTimer.elapsed() >= kCheckpointSeconds * 1000)) {
                checkpointSequence_ = store_.maxVectorSequence();
                saveCheckpoint();
                vectorsSinceCheckpoint = 0;
                checkpointTimer.restart();
            }
            if (cancelRequested_.load(std::memory_order_relaxed)) {
                checkpointSequence_ = store_.maxVectorSequence();
                saveCheckpoint();
                store_.refreshActiveState();
                busy_ = false;
                Q_EMIT buildCompleted(summary(), true);
                return;
            }
        }

        store_.finishScan(scanId);
        checkpointSequence_ = store_.maxVectorSequence();
        saveCheckpoint();
        compactIfNeeded();
        store_.refreshActiveState();
        if (cancelRequested_.load(std::memory_order_relaxed)) {
            busy_ = false;
            Q_EMIT buildCompleted(summary(), true);
            return;
        }
        busy_ = false;
        Q_EMIT buildCompleted(summary(), false);
    } catch (const std::exception& error) {
        store_.rollbackWriteBatch();
        busy_ = false;
        Q_EMIT failed(QStringLiteral("Build failed"), QString::fromUtf8(error.what()));
    }
}

void SimilarityService::saveCheckpoint() {
    if (!index_ || store_.isReadOnly()) {
        return;
    }
    const QString fileName = QStringLiteral("index_%1_%2.faiss")
                                 .arg(checkpointSequence_)
                                 .arg(QDateTime::currentMSecsSinceEpoch());
    const QString path = QDir(store_.catalogDirectory()).filePath(fileName);
    index_->saveAtomic(path);
    store_.commitCheckpoint(path, checkpointSequence_);
    store_.removeObsoleteIndexes(path);
}

bool SimilarityService::compactIfNeeded() {
    const std::int64_t total = store_.totalVectorCount();
    const std::int64_t active = store_.activeVectorCount();
    if (total == 0 || static_cast<double>(total - active) / static_cast<double>(total) < kCompactionRatio) {
        return false;
    }

    auto compacted = std::make_unique<FaissIndex>();
    compacted->create();
    std::int64_t cursor = 0;
    std::int64_t processed = 0;
    while (true) {
        if (cancelRequested_.load(std::memory_order_relaxed)) {
            return false;
        }
        const QVector<StoredFeature> rows = store_.activeFeatureBatch(cursor, 4096);
        if (rows.isEmpty()) {
            break;
        }
        std::vector<std::int64_t> ids;
        std::vector<std::vector<float>> embeddings;
        ids.reserve(rows.size());
        embeddings.reserve(rows.size());
        for (const StoredFeature& row : rows) {
            cursor = row.sequence;
            ids.push_back(row.vectorId);
            embeddings.push_back(row.embedding);
        }
        compacted->add(ids, embeddings);
        processed += rows.size();
        BuildProgress progress;
        progress.stage = QStringLiteral("Optimizing index");
        progress.processed = processed;
        progress.total = active;
        Q_EMIT buildProgress(progress);
    }
    index_ = std::move(compacted);
    checkpointSequence_ = store_.maxVectorSequence();
    store_.compactFeatureLog();
    saveCheckpoint();
    return true;
}

void SimilarityService::search(const QString& queryPath, SearchOptions options) {
    if (busy_ || !index_ || index_->empty()) {
        Q_EMIT failed(QStringLiteral("Search unavailable"), QStringLiteral("Build a catalog first."));
        return;
    }
    options.minSimilarity = std::clamp(options.minSimilarity, 0.0F, 1.0F);
    options.topNum = std::clamp(options.topNum, 1, 1000);
    busy_ = true;
    QElapsedTimer timer;
    timer.start();
    try {
        PreparedImage query = preprocessor_.prepare(queryPath);
        const auto embeddings = engine_.embed(query.tensors);
        Q_EMIT runtimeReady(engine_.providerName());
        const int candidates = std::max(800, options.topNum * 4);
        const int efSearch = std::max(1024, candidates);
        std::vector<VectorScore> rawCandidates;
        rawCandidates.reserve(static_cast<std::size_t>(candidates) * embeddings.size());
        for (const auto& embedding : embeddings) {
            const auto matches = index_->search(embedding, candidates, efSearch);
            for (const auto& [vectorId, score] : matches) {
                rawCandidates.push_back({vectorId, score});
            }
        }
        const auto ranked = rankUniqueImages(
            rawCandidates,
            options.minSimilarity,
            options.topNum,
            [this](const std::int64_t vectorId) { return store_.isVectorActive(vectorId); });

        QVector<std::int64_t> imageIds;
        imageIds.reserve(static_cast<qsizetype>(ranked.size()));
        for (const auto& [imageId, score] : ranked) {
            Q_UNUSED(score)
            imageIds.push_back(imageId);
        }
        const auto metadata = store_.imageMetadata(imageIds);
        QVector<SearchResult> results;
        results.reserve(static_cast<qsizetype>(ranked.size()));
        for (const auto& [imageId, score] : ranked) {
            const auto iterator = metadata.constFind(imageId);
            if (iterator == metadata.constEnd()) {
                continue;
            }
            SearchResult result;
            result.imageId = imageId;
            result.relativePath = iterator->relativePath;
            result.absolutePath = QDir(store_.sourceDirectory()).filePath(result.relativePath);
            result.score = score;
            result.width = iterator->width;
            result.height = iterator->height;
            results.push_back(std::move(result));
        }
        busy_ = false;
        Q_EMIT searchCompleted(queryPath, results, timer.elapsed());
    } catch (const std::exception& error) {
        busy_ = false;
        Q_EMIT failed(QStringLiteral("Search failed"), QString::fromUtf8(error.what()));
    }
}

CatalogSummary SimilarityService::summary() const {
    CatalogSummary result;
    result.sourceDirectory = store_.sourceDirectory();
    result.imageCount = store_.isOpen() ? store_.activeImageCount() : 0;
    result.activeVectorCount = store_.isOpen() ? store_.activeVectorCount() : 0;
    result.readOnly = store_.isReadOnly();
    return result;
}

}  // namespace sf
