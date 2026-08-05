#include "app/main_window.h"

#include "core/app_types.h"

#include <QApplication>
#include <QCoreApplication>
#include <QFont>

int main(int argc, char* argv[]) {
    QApplication application(argc, argv);
    QCoreApplication::setOrganizationName(QStringLiteral("SimilarFinder"));
    QCoreApplication::setApplicationName(QStringLiteral("SimilarFinder"));
    QCoreApplication::setApplicationVersion(QStringLiteral(SIMILARFINDER_VERSION));

    qRegisterMetaType<sf::SearchOptions>();
    qRegisterMetaType<QVector<sf::SearchResult>>();
    qRegisterMetaType<sf::BuildProgress>();
    qRegisterMetaType<sf::CatalogSummary>();

    QFont font(QStringLiteral("Segoe UI"));
    font.setPointSize(10);
    application.setFont(font);

    MainWindow window;
    window.show();
    return QApplication::exec();
}
