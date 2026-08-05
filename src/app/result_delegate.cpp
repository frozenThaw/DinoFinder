#include "app/result_delegate.h"

#include "app/result_model.h"

#include <QApplication>
#include <QImage>
#include <QPainter>
#include <QStyle>
#include <QTextOption>

ResultDelegate::ResultDelegate(QObject* parent) : QStyledItemDelegate(parent) {}

void ResultDelegate::paint(
    QPainter* painter,
    const QStyleOptionViewItem& option,
    const QModelIndex& index) const {
    painter->save();
    painter->setRenderHint(QPainter::Antialiasing, true);
    const QRect card = option.rect.adjusted(4, 4, -4, -4);
    const bool selected = (option.state & QStyle::State_Selected) != 0;
    painter->setPen(selected ? QColor(QStringLiteral("#087f5b")) : QColor(QStringLiteral("#dfe3e6")));
    painter->setBrush(selected ? QColor(QStringLiteral("#eef8f4")) : Qt::white);
    painter->drawRoundedRect(card, 5, 5);

    const QRect imageArea(card.left() + 12, card.top() + 12, card.width() - 24, 158);
    painter->fillRect(imageArea, QColor(QStringLiteral("#f3f5f5")));
    const QImage thumbnail = qvariant_cast<QImage>(index.data(Qt::DecorationRole));
    if (!thumbnail.isNull()) {
        const QSize size = thumbnail.size().scaled(imageArea.size() - QSize(10, 10), Qt::KeepAspectRatio);
        const QRect target(QPoint(0, 0), size);
        const QRect centered = target.translated(imageArea.center() - target.center());
        painter->drawImage(centered, thumbnail);
    }

    const float score = index.data(ResultModel::ScoreRole).toFloat();
    QRect scoreRect(card.left() + 12, imageArea.bottom() + 10, card.width() - 24, 22);
    QFont scoreFont = option.font;
    scoreFont.setBold(true);
    scoreFont.setPointSize(11);
    painter->setFont(scoreFont);
    painter->setPen(QColor(QStringLiteral("#087f5b")));
    painter->drawText(scoreRect, Qt::AlignLeft | Qt::AlignVCenter, QString::number(score, 'f', 4));

    QRect nameRect(card.left() + 12, scoreRect.bottom() + 3, card.width() - 24, 20);
    QFont nameFont = option.font;
    nameFont.setBold(true);
    painter->setFont(nameFont);
    painter->setPen(QColor(QStringLiteral("#1b1f22")));
    const QString name = index.data(Qt::DisplayRole).toString();
    painter->drawText(
        nameRect,
        Qt::AlignLeft | Qt::AlignVCenter,
        option.fontMetrics.elidedText(name, Qt::ElideMiddle, nameRect.width()));

    QRect detailRect(card.left() + 12, nameRect.bottom() + 1, card.width() - 24, 18);
    painter->setFont(option.font);
    painter->setPen(QColor(QStringLiteral("#687077")));
    const QString relativePath = index.data(ResultModel::RelativePathRole).toString();
    painter->drawText(
        detailRect,
        Qt::AlignLeft | Qt::AlignVCenter,
        option.fontMetrics.elidedText(relativePath, Qt::ElideMiddle, detailRect.width()));
    painter->restore();
}

QSize ResultDelegate::sizeHint(
    const QStyleOptionViewItem& option,
    const QModelIndex& index) const {
    Q_UNUSED(option)
    Q_UNUSED(index)
    return {218, 252};
}
