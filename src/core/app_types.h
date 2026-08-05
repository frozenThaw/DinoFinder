#pragma once

#include <QMetaType>
#include <QString>
#include <QVector>

#include <cstdint>
#include <vector>

namespace sf {

inline constexpr int kImageSize = 224;
inline constexpr int kEmbeddingDimension = 384;
inline constexpr int kHnswLinks = 32;
inline constexpr int kHnswEfConstruction = 200;
inline constexpr int kCheckpointVectorCount = 5000;
inline constexpr int kCheckpointSeconds = 30;
inline constexpr double kCompactionRatio = 0.15;
inline constexpr int kPreprocessVersion = 4;

enum class BackgroundVariant : std::uint8_t {
    Original = 0,
    White = 1,
    Gray = 2,
    Black = 3,
};

struct ImageTensor {
    BackgroundVariant variant = BackgroundVariant::Original;
    std::vector<float> nchw;
};

struct PreparedImage {
    int width = 0;
    int height = 0;
    bool hasTransparency = false;
    std::vector<ImageTensor> tensors;
};

struct SearchOptions {
    float minSimilarity = 0.80F;
    int topNum = 200;
};

struct SearchResult {
    std::int64_t imageId = 0;
    QString absolutePath;
    QString relativePath;
    float score = 0.0F;
    int width = 0;
    int height = 0;
};

struct BuildProgress {
    QString stage;
    std::int64_t processed = 0;
    std::int64_t total = 0;
    double itemsPerSecond = 0.0;
    qint64 etaSeconds = -1;
};

struct CatalogSummary {
    QString sourceDirectory;
    std::int64_t imageCount = 0;
    std::int64_t activeVectorCount = 0;
    bool readOnly = false;
};

}  // namespace sf

Q_DECLARE_METATYPE(sf::SearchOptions)
Q_DECLARE_METATYPE(sf::SearchResult)
Q_DECLARE_METATYPE(QVector<sf::SearchResult>)
Q_DECLARE_METATYPE(sf::BuildProgress)
Q_DECLARE_METATYPE(sf::CatalogSummary)
