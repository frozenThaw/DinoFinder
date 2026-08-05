#pragma once

#include <QStyledItemDelegate>

class ResultDelegate final : public QStyledItemDelegate {
    Q_OBJECT

public:
    explicit ResultDelegate(QObject* parent = nullptr);

    void paint(
        QPainter* painter,
        const QStyleOptionViewItem& option,
        const QModelIndex& index) const override;
    [[nodiscard]] QSize sizeHint(
        const QStyleOptionViewItem& option,
        const QModelIndex& index) const override;
};
