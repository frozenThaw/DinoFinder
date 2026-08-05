#include "core/catalog_store.h"
#include "core/dino_engine.h"
#include "core/faiss_index.h"
#include "core/feature_utils.h"
#include "core/image_preprocessor.h"
#include "core/similarity_service.h"

#include <QDir>
#include <QFileInfo>
#include <QImage>
#include <QStandardPaths>
#include <QTemporaryDir>
#include <QTest>
#include <QUuid>

#include <cmath>
#include <vector>

class CoreTests final : public QObject {
  Q_OBJECT

private Q_SLOTS:
  void initTestCase();
  void normalizeAndHalfRoundTrip();
  void vectorIdRoundTrip();
  void rankFiltersDeduplicatesAndCaps();
  void preprocessesOpaqueAndTransparentImages();
  void dinoEngineEmbedsImage();
  void hnswReturnsNearestVector();
  void catalogTracksFeatureGenerationsAndDeletion();
  void endToEndCatalogSearch();
};

void CoreTests::initTestCase() { QStandardPaths::setTestModeEnabled(true); }

void CoreTests::normalizeAndHalfRoundTrip() {
  std::vector<float> values = {3.0F, 4.0F};
  sf::normalizeL2(values);
  QVERIFY(std::abs(values[0] - 0.6F) < 1.0e-6F);
  QVERIFY(std::abs(values[1] - 0.8F) < 1.0e-6F);

  const auto halves = sf::toHalf(values);
  const auto restored = sf::fromHalf(halves);
  QCOMPARE(restored.size(), values.size());
  QVERIFY(std::abs(restored[0] - values[0]) < 0.001F);
  QVERIFY(std::abs(restored[1] - values[1]) < 0.001F);
}

void CoreTests::vectorIdRoundTrip() {
  const std::int64_t vectorId =
      sf::encodeVectorId(987654, 42, sf::BackgroundVariant::Black);
  QCOMPARE(sf::imageIdFromVectorId(vectorId), 987654);
  QCOMPARE(sf::generationFromVectorId(vectorId), 42);
  QCOMPARE(sf::variantFromVectorId(vectorId), sf::BackgroundVariant::Black);
}

void CoreTests::rankFiltersDeduplicatesAndCaps() {
  const std::int64_t imageOneWhite =
      sf::encodeVectorId(1, 1, sf::BackgroundVariant::White);
  const std::int64_t imageOneBlack =
      sf::encodeVectorId(1, 1, sf::BackgroundVariant::Black);
  const std::int64_t imageTwo =
      sf::encodeVectorId(2, 1, sf::BackgroundVariant::Original);
  const std::int64_t inactive =
      sf::encodeVectorId(3, 1, sf::BackgroundVariant::Original);
  const std::vector<sf::VectorScore> candidates = {
      {imageOneWhite, 0.81F},
      {imageOneBlack, 0.91F},
      {imageTwo, 0.80F},
      {inactive, 0.99F},
  };
  const auto ranked = sf::rankUniqueImages(
      candidates, 0.80F, 2,
      [inactive](const std::int64_t id) { return id != inactive; });
  QCOMPARE(ranked.size(), 2ULL);
  QCOMPARE(ranked[0].first, 1);
  QVERIFY(std::abs(ranked[0].second - 0.91F) < 1.0e-6F);
  QCOMPARE(ranked[1].first, 2);
}

void CoreTests::preprocessesOpaqueAndTransparentImages() {
  QTemporaryDir directory;
  QVERIFY(directory.isValid());
  const QString opaquePath =
      QDir(directory.path()).filePath(QStringLiteral("opaque.png"));
  const QString alphaPath =
      QDir(directory.path()).filePath(QStringLiteral("透明图.png"));
  QImage opaque(24, 8, QImage::Format_RGB888);
  opaque.fill(QColor(20, 40, 60));
  QVERIFY(opaque.save(opaquePath));
  QImage alpha(24, 8, QImage::Format_RGBA8888);
  alpha.fill(Qt::transparent);
  alpha.setPixelColor(12, 4, QColor(255, 0, 0, 255));
  QVERIFY(alpha.save(alphaPath));

  sf::ImagePreprocessor preprocessor;
  const sf::PreparedImage opaqueResult = preprocessor.prepare(opaquePath);
  const sf::PreparedImage alphaResult = preprocessor.prepare(alphaPath);
  QCOMPARE(opaqueResult.tensors.size(), 1ULL);
  QCOMPARE(alphaResult.tensors.size(), 3ULL);
  QCOMPARE(alphaResult.width, 24);
  QCOMPARE(alphaResult.height, 8);
  QVERIFY(alphaResult.hasTransparency);
  for (const sf::ImageTensor &tensor : alphaResult.tensors) {
    QCOMPARE(tensor.nchw.size(), 3ULL * sf::kImageSize * sf::kImageSize);
  }
}

void CoreTests::dinoEngineEmbedsImage() {
  QTemporaryDir directory;
  QVERIFY(directory.isValid());
  const QString imagePath =
      QDir(directory.path()).filePath(QStringLiteral("query.png"));
  QImage image(224, 160, QImage::Format_RGB888);
  image.fill(QColor(38, 121, 204));
  QVERIFY(image.save(imagePath));

  sf::ImagePreprocessor preprocessor;
  const sf::PreparedImage prepared = preprocessor.prepare(imagePath);
  const QString modelDirectory = QDir(QStringLiteral(QT_TESTCASE_SOURCEDIR))
                                     .filePath(QStringLiteral("models"));
  sf::DinoEngine engine;
  QString error;
  QVERIFY2(
      engine.initialize(
          QDir(modelDirectory).filePath(QStringLiteral("dinov2_vits14.onnx")),
          QDir(modelDirectory)
              .filePath(QStringLiteral("dinov2_vits14_manifest.json")),
          false, &error),
      qPrintable(error));
  const auto embeddings = engine.embed(prepared.tensors);
  QCOMPARE(embeddings.size(), 1ULL);
  QCOMPARE(embeddings[0].size(),
           static_cast<std::size_t>(sf::kEmbeddingDimension));
  double squaredNorm = 0.0;
  for (const float value : embeddings[0]) {
    squaredNorm += static_cast<double>(value) * value;
  }
  QVERIFY(std::abs(std::sqrt(squaredNorm) - 1.0) < 1.0e-5);
}

void CoreTests::hnswReturnsNearestVector() {
  sf::FaissIndex index;
  index.create();
  std::vector<std::int64_t> ids;
  std::vector<std::vector<float>> vectors;
  for (int item = 0; item < 8; ++item) {
    ids.push_back(
        sf::encodeVectorId(item + 1, 1, sf::BackgroundVariant::Original));
    std::vector<float> embedding(sf::kEmbeddingDimension, 0.0F);
    embedding[static_cast<std::size_t>(item)] = 1.0F;
    vectors.push_back(std::move(embedding));
  }
  index.add(ids, vectors);
  const auto result = index.search(vectors[5], 3, 64);
  QVERIFY(!result.empty());
  QCOMPARE(result[0].first, ids[5]);
  QVERIFY(result[0].second > 0.999F);
}

void CoreTests::catalogTracksFeatureGenerationsAndDeletion() {
  try {
    QTemporaryDir source;
    QTemporaryDir dataRoot;
    QVERIFY(source.isValid());
    QVERIFY(dataRoot.isValid());
    QVERIFY(qputenv("SIMILARFINDER_DATA_ROOT", dataRoot.path().toUtf8()));
    sf::CatalogStore store;
    store.open(source.path(), QUuid::createUuid()
                                  .toString(QUuid::WithoutBraces)
                                  .remove(QLatin1Char('-'))
                                  .leftJustified(64, QLatin1Char('a')));
    const QString catalogPath = store.catalogDirectory();
    const std::int64_t scanOne = store.beginScan();
    const sf::FileDecision first =
        store.inspectFile(QStringLiteral("asset.png"), 100, 1000, scanOne);
    QVERIFY(first.requiresFeatures);
    std::vector<float> embedding(sf::kEmbeddingDimension, 0.0F);
    embedding[0] = 1.0F;
    store.beginWriteBatch();
    const auto firstIds = store.commitImageFeatures(
        first, 100, 1000, 32, 32, false, {sf::BackgroundVariant::Original},
        {embedding});
    store.commitWriteBatch();
    store.finishScan(scanOne);
    QVERIFY(store.isVectorActive(firstIds[0]));

    const std::int64_t scanTwo = store.beginScan();
    const sf::FileDecision modified =
        store.inspectFile(QStringLiteral("asset.png"), 101, 2000, scanTwo);
    QVERIFY(modified.requiresFeatures);
    store.beginWriteBatch();
    const auto secondIds = store.commitImageFeatures(
        modified, 101, 2000, 32, 32, false, {sf::BackgroundVariant::Original},
        {embedding});
    store.commitWriteBatch();
    store.finishScan(scanTwo);
    store.refreshActiveState();
    QVERIFY(!store.isVectorActive(firstIds[0]));
    QVERIFY(store.isVectorActive(secondIds[0]));

    const std::int64_t scanThree = store.beginScan();
    store.finishScan(scanThree);
    QVERIFY(!store.isVectorActive(secondIds[0]));
    store.close();
    QDir(catalogPath).removeRecursively();
    qunsetenv("SIMILARFINDER_DATA_ROOT");
  } catch (const std::exception &error) {
    qunsetenv("SIMILARFINDER_DATA_ROOT");
    QFAIL(error.what());
  }
}

void CoreTests::endToEndCatalogSearch() {
  const QString sourceDirectory =
      qEnvironmentVariable("SIMILARFINDER_INTEGRATION_SOURCE");
  const QString queryPath =
      qEnvironmentVariable("SIMILARFINDER_INTEGRATION_QUERY");
  if (sourceDirectory.isEmpty() || queryPath.isEmpty()) {
    QSKIP("Set SIMILARFINDER_INTEGRATION_SOURCE and "
          "SIMILARFINDER_INTEGRATION_QUERY to run.");
  }

  QTemporaryDir dataRoot;
  QVERIFY(dataRoot.isValid());
  QVERIFY(qputenv("SIMILARFINDER_DATA_ROOT", dataRoot.path().toUtf8()));

  sf::SimilarityService service;
  QString failure;
  QString provider;
  sf::CatalogSummary builtSummary;
  QVector<sf::SearchResult> results;
  connect(&service, &sf::SimilarityService::failed, this,
          [&failure](const QString &title, const QString &message) {
            failure = title + QStringLiteral(": ") + message;
          });
  connect(&service, &sf::SimilarityService::runtimeReady, this,
          [&provider](const QString &value) { provider = value; });
  connect(
      &service, &sf::SimilarityService::buildCompleted, this,
      [&builtSummary](const sf::CatalogSummary &summary, const bool cancelled) {
        if (!cancelled) {
          builtSummary = summary;
        }
      });
  connect(&service, &sf::SimilarityService::searchCompleted, this,
          [&results](const QString &, const QVector<sf::SearchResult> &value,
                     const qint64) { results = value; });

  service.initialize(QStringLiteral(QT_TESTCASE_SOURCEDIR));
  service.openDirectory(sourceDirectory);
  service.buildCatalog();
  sf::SearchOptions options;
  options.minSimilarity = 0.0F;
  options.topNum = 5;
  service.search(queryPath, options);

  qunsetenv("SIMILARFINDER_DATA_ROOT");
  QVERIFY2(failure.isEmpty(), qPrintable(failure));
  QVERIFY(provider == QStringLiteral("CUDA") ||
          provider == QStringLiteral("CPU"));
  QVERIFY(builtSummary.imageCount > 0);
  QVERIFY(builtSummary.activeVectorCount >= builtSummary.imageCount);
  QVERIFY(!results.isEmpty());
    for (int index = 1; index < results.size(); ++index) {
        QVERIFY(results[index - 1].score >= results[index].score);
    }
    for (int index = 0; index < results.size(); ++index) {
        qInfo("Result %d: %.6f %s",
            index + 1,
            static_cast<double>(results[index].score),
            qPrintable(results[index].relativePath));
    }
    qInfo("Integration provider: %s, images: %lld, vectors: %lld, results: %lld",
        qPrintable(provider), static_cast<long long>(builtSummary.imageCount),
        static_cast<long long>(builtSummary.activeVectorCount),
        static_cast<long long>(results.size()));
}

QTEST_GUILESS_MAIN(CoreTests)
#include "test_core.moc"
