#pragma once

#include "core/app_types.h"

#include <QBitArray>
#include <QHash>
#include <QLockFile>
#include <QSqlDatabase>
#include <QString>
#include <QVector>

#include <cstdint>
#include <memory>
#include <vector>

class QFile;

namespace sf {

struct FileDecision {
    std::int64_t imageId = 0;
    QString relativePath;
    std::uint16_t generation = 0;
    bool requiresFeatures = false;
};

struct StoredFeature {
    std::int64_t sequence = 0;
    std::int64_t vectorId = 0;
    std::vector<float> embedding;
};

struct ImageMetadata {
    std::int64_t imageId = 0;
    QString relativePath;
    int width = 0;
    int height = 0;
};

struct IndexCheckpoint {
    QString path;
    std::int64_t maxVectorSequence = 0;
};

class CatalogStore final {
public:
    CatalogStore();
    ~CatalogStore();

    CatalogStore(const CatalogStore&) = delete;
    CatalogStore& operator=(const CatalogStore&) = delete;

    void open(const QString& sourceDirectory, const QString& modelHash);
    void close();

    [[nodiscard]] bool isOpen() const;
    [[nodiscard]] bool isReadOnly() const;
    [[nodiscard]] QString sourceDirectory() const;
    [[nodiscard]] QString catalogDirectory() const;

    std::int64_t beginScan();
    FileDecision inspectFile(
        const QString& relativePath,
        std::int64_t fileSize,
        std::int64_t modifiedMilliseconds,
        std::int64_t scanId);
    void finishScan(std::int64_t scanId);

    void beginWriteBatch();
    std::vector<std::int64_t> commitImageFeatures(
        const FileDecision& decision,
        std::int64_t fileSize,
        std::int64_t modifiedMilliseconds,
        int width,
        int height,
        bool hasTransparency,
        const std::vector<BackgroundVariant>& variants,
        const std::vector<std::vector<float>>& embeddings);
    void recordImageError(const FileDecision& decision, const QString& error);
    void commitWriteBatch();
    void rollbackWriteBatch();

    [[nodiscard]] QVector<StoredFeature> activeFeatureBatch(
        std::int64_t afterSequence,
        int limit) const;
    [[nodiscard]] std::int64_t maxVectorSequence() const;
    [[nodiscard]] std::int64_t activeImageCount() const;
    [[nodiscard]] std::int64_t activeVectorCount() const;
    [[nodiscard]] std::int64_t totalVectorCount() const;

    void refreshActiveState();
    [[nodiscard]] bool isVectorActive(std::int64_t vectorId) const;
    [[nodiscard]] QHash<std::int64_t, ImageMetadata> imageMetadata(
        const QVector<std::int64_t>& imageIds) const;

    [[nodiscard]] IndexCheckpoint checkpoint() const;
    void commitCheckpoint(const QString& indexPath, std::int64_t maxVectorSequence);
    void removeObsoleteIndexes(const QString& keepPath) const;
    void compactFeatureLog();

private:
    std::int64_t appendFeature(std::int64_t vectorId, const std::vector<float>& embedding);
    std::vector<float> readFeature(std::int64_t offset) const;
    void createSchema();
    void ensureWritable() const;

    QString connectionName_;
    QSqlDatabase database_;
    QString sourceDirectory_;
    QString catalogDirectory_;
    QString featuresPath_;
    std::unique_ptr<QLockFile> writeLock_;
    bool readOnly_ = false;
    bool writeBatchActive_ = false;
    std::unique_ptr<QFile> featureWriteFile_;
    std::vector<std::uint16_t> activeGenerations_;
    QBitArray activeImages_;
};

}  // namespace sf
