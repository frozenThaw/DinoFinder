#pragma once

#include "core/app_types.h"

#include <QAbstractListModel>
#include <QCache>
#include <QImage>
#include <QSet>

class ResultModel final : public QAbstractListModel {
    Q_OBJECT

public:
    enum Role {
        AbsolutePathRole = Qt::UserRole + 1,
        RelativePathRole,
        ScoreRole,
        DimensionsRole,
    };

    explicit ResultModel(QObject* parent = nullptr);

    [[nodiscard]] int rowCount(const QModelIndex& parent = {}) const override;
    [[nodiscard]] QVariant data(const QModelIndex& index, int role) const override;
    [[nodiscard]] QHash<int, QByteArray> roleNames() const override;

    void setResults(const QVector<sf::SearchResult>& results);
    [[nodiscard]] sf::SearchResult resultAt(int row) const;

private:
    void requestThumbnail(int row, const QString& path) const;
    void acceptThumbnail(int row, const QString& path, const QImage& image);
    void trimDiskCache();

    QVector<sf::SearchResult> results_;
    mutable QCache<QString, QImage> thumbnails_{256};
    mutable QSet<QString> loading_;
    QString cacheDirectory_;
};
