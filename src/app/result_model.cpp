#include "app/result_model.h"

#include <QCryptographicHash>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QImageReader>
#include <QMetaObject>
#include <QPointer>
#include <QStandardPaths>
#include <QThreadPool>

#include <algorithm>

namespace {

QString thumbnailKey(const QString& path) {
    const QFileInfo info(path);
    const QByteArray identity = (path.toCaseFolded() + QLatin1Char('|')
                                 + QString::number(info.size()) + QLatin1Char('|')
                                 + QString::number(info.lastModified().toMSecsSinceEpoch()))
                                    .toUtf8();
    return QString::fromLatin1(QCryptographicHash::hash(identity, QCryptographicHash::Sha256).toHex());
}

QImage loadThumbnail(const QString& sourcePath, const QString& cachePath) {
    QImage cached(cachePath);
    if (!cached.isNull()) {
        QFile(cachePath).setFileTime(QDateTime::currentDateTimeUtc(), QFileDevice::FileAccessTime);
        return cached;
    }
    QImageReader reader(sourcePath);
    reader.setAutoTransform(true);
    const QSize original = reader.size();
    if (original.isValid()) {
        reader.setScaledSize(original.scaled(QSize(176, 164), Qt::KeepAspectRatio));
    }
    QImage image = reader.read();
    if (!image.isNull()) {
        image.save(cachePath, "PNG");
    }
    return image;
}

}  // namespace

ResultModel::ResultModel(QObject* parent) : QAbstractListModel(parent) {
    cacheDirectory_ = QDir(QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation))
                          .filePath(QStringLiteral("thumbnails"));
    QDir().mkpath(cacheDirectory_);
}

int ResultModel::rowCount(const QModelIndex& parent) const {
    return parent.isValid() ? 0 : results_.size();
}

QVariant ResultModel::data(const QModelIndex& index, const int role) const {
    if (!index.isValid() || index.row() < 0 || index.row() >= results_.size()) {
        return {};
    }
    const sf::SearchResult& result = results_[index.row()];
    switch (role) {
    case Qt::DisplayRole:
        return QFileInfo(result.absolutePath).fileName();
    case Qt::DecorationRole: {
        if (const QImage* image = thumbnails_.object(result.absolutePath)) {
            return *image;
        }
        requestThumbnail(index.row(), result.absolutePath);
        return {};
    }
    case AbsolutePathRole:
        return result.absolutePath;
    case RelativePathRole:
        return result.relativePath;
    case ScoreRole:
        return result.score;
    case DimensionsRole:
        return QStringLiteral("%1 × %2").arg(result.width).arg(result.height);
    default:
        return {};
    }
}

QHash<int, QByteArray> ResultModel::roleNames() const {
    return {
        {AbsolutePathRole, "absolutePath"},
        {RelativePathRole, "relativePath"},
        {ScoreRole, "score"},
        {DimensionsRole, "dimensions"},
    };
}

void ResultModel::setResults(const QVector<sf::SearchResult>& results) {
    beginResetModel();
    results_ = results;
    loading_.clear();
    endResetModel();
    trimDiskCache();
}

sf::SearchResult ResultModel::resultAt(const int row) const {
    return row >= 0 && row < results_.size() ? results_[row] : sf::SearchResult{};
}

void ResultModel::requestThumbnail(const int row, const QString& path) const {
    if (loading_.contains(path)) {
        return;
    }
    loading_.insert(path);
    const QString cachePath = QDir(cacheDirectory_).filePath(thumbnailKey(path) + QStringLiteral(".png"));
    QPointer<ResultModel> self(const_cast<ResultModel*>(this));
    QThreadPool::globalInstance()->start([self, row, path, cachePath]() {
        const QImage image = loadThumbnail(path, cachePath);
        if (self) {
            QMetaObject::invokeMethod(self, [self, row, path, image]() {
                if (self) {
                    self->acceptThumbnail(row, path, image);
                }
            });
        }
    });
}

void ResultModel::acceptThumbnail(
    const int row,
    const QString& path,
    const QImage& image) {
    loading_.remove(path);
    if (!image.isNull()) {
        thumbnails_.insert(path, new QImage(image));
    }
    if (row >= 0 && row < results_.size() && results_[row].absolutePath == path) {
        const QModelIndex changed = index(row);
        Q_EMIT dataChanged(changed, changed, {Qt::DecorationRole});
    }
}

void ResultModel::trimDiskCache() {
    constexpr qint64 maximumBytes = 1024LL * 1024LL * 1024LL;
    QDir directory(cacheDirectory_);
    QFileInfoList files = directory.entryInfoList(
        {QStringLiteral("*.png")}, QDir::Files, QDir::Time | QDir::Reversed);
    qint64 total = 0;
    for (const QFileInfo& file : files) {
        total += file.size();
    }
    for (const QFileInfo& file : files) {
        if (total <= maximumBytes) {
            break;
        }
        total -= file.size();
        QFile::remove(file.absoluteFilePath());
    }
}
