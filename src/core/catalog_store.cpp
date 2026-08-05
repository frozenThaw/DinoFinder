#include "core/catalog_store.h"

#include "core/feature_utils.h"

#include <QCryptographicHash>
#include <QDataStream>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSaveFile>
#include <QSqlError>
#include <QSqlQuery>
#include <QStandardPaths>
#include <QtEndian>
#include <QUuid>

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace sf {

namespace {

constexpr std::uint32_t kFeatureMagic = 0x31464653U;  // SFF1
constexpr std::uint16_t kFeatureFormatVersion = 1;
[[noreturn]] void throwSql(const QString& operation, const QSqlError& error) {
    throw std::runtime_error((operation + QStringLiteral(": ") + error.text()).toStdString());
}

void executeOrThrow(QSqlQuery& query, const QString& operation) {
    if (!query.exec()) {
        throwSql(operation, query.lastError());
    }
}

std::int64_t writeFeatureRecord(
    QIODevice& device,
    const std::int64_t vectorId,
    const std::vector<float>& embedding) {
    if (embedding.size() != kEmbeddingDimension) {
        throw std::invalid_argument("Feature log embedding has an invalid dimension");
    }
    const std::vector<std::uint16_t> halves = toHalf(embedding);
    QByteArray payload(static_cast<qsizetype>(halves.size() * sizeof(std::uint16_t)), Qt::Uninitialized);
    for (std::size_t index = 0; index < halves.size(); ++index) {
        const std::uint16_t littleEndian = qToLittleEndian(halves[index]);
        std::memcpy(payload.data() + index * sizeof(std::uint16_t), &littleEndian, sizeof(littleEndian));
    }
    const std::int64_t offset = device.pos();
    QDataStream stream(&device);
    stream.setByteOrder(QDataStream::LittleEndian);
    stream << kFeatureMagic << kFeatureFormatVersion << static_cast<std::uint16_t>(kEmbeddingDimension)
           << static_cast<qint64>(vectorId) << crc32(payload);
    if (stream.status() != QDataStream::Ok || device.write(payload) != payload.size()) {
        throw std::runtime_error("Cannot write feature log record");
    }
    return offset;
}

std::vector<float> readFeatureRecord(QIODevice& device, const std::int64_t offset) {
    if (!device.seek(offset)) {
        throw std::runtime_error("Cannot seek feature log record");
    }
    QDataStream stream(&device);
    stream.setByteOrder(QDataStream::LittleEndian);
    std::uint32_t magic = 0;
    std::uint16_t version = 0;
    std::uint16_t dimension = 0;
    qint64 vectorId = 0;
    std::uint32_t expectedCrc = 0;
    stream >> magic >> version >> dimension >> vectorId >> expectedCrc;
    Q_UNUSED(vectorId)
    if (stream.status() != QDataStream::Ok || magic != kFeatureMagic
        || version != kFeatureFormatVersion || dimension != kEmbeddingDimension) {
        throw std::runtime_error("Feature log record header is invalid");
    }
    const QByteArray payload = device.read(kEmbeddingDimension * static_cast<qint64>(sizeof(std::uint16_t)));
    if (payload.size() != kEmbeddingDimension * static_cast<qsizetype>(sizeof(std::uint16_t))
        || crc32(payload) != expectedCrc) {
        throw std::runtime_error("Feature log record checksum mismatch");
    }
    std::vector<std::uint16_t> halves(kEmbeddingDimension);
    for (int index = 0; index < kEmbeddingDimension; ++index) {
        std::uint16_t littleEndian = 0;
        std::memcpy(&littleEndian, payload.constData() + index * sizeof(std::uint16_t), sizeof(littleEndian));
        halves[index] = qFromLittleEndian(littleEndian);
    }
    auto embedding = fromHalf(halves);
    normalizeL2(embedding);
    return embedding;
}

QString normalizedSourcePath(const QString& path) {
    QFileInfo info(path);
    QString normalized = info.canonicalFilePath();
    if (normalized.isEmpty()) {
        normalized = info.absoluteFilePath();
    }
    return QDir::cleanPath(normalized);
}

QString catalogKey(const QString& sourceDirectory, const QString& modelHash) {
    const QByteArray identity = (sourceDirectory.toCaseFolded() + QLatin1Char('|') + modelHash
                                 + QLatin1Char('|') + QString::number(kPreprocessVersion))
                                    .toUtf8();
    return QString::fromLatin1(QCryptographicHash::hash(identity, QCryptographicHash::Sha256).toHex().left(24));
}

}  // namespace

CatalogStore::CatalogStore()
    : connectionName_(QStringLiteral("similarfinder-%1").arg(QUuid::createUuid().toString(QUuid::WithoutBraces))) {}

CatalogStore::~CatalogStore() {
    close();
}

void CatalogStore::open(const QString& sourceDirectory, const QString& modelHash) {
    close();
    sourceDirectory_ = normalizedSourcePath(sourceDirectory);
    if (!QFileInfo(sourceDirectory_).isDir()) {
        throw std::runtime_error(("Source directory does not exist: " + sourceDirectory_).toStdString());
    }

    QString appData = qEnvironmentVariable("SIMILARFINDER_DATA_ROOT");
    if (appData.isEmpty()) {
        appData = QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation);
    }
    catalogDirectory_ = QDir(appData).filePath(
        QStringLiteral("catalogs/%1").arg(catalogKey(sourceDirectory_, modelHash)));
    if (!QDir().mkpath(catalogDirectory_)) {
        throw std::runtime_error(("Cannot create catalog directory: " + catalogDirectory_).toStdString());
    }
    featuresPath_ = QDir(catalogDirectory_).filePath(QStringLiteral("features.fp16"));

    writeLock_ = std::make_unique<QLockFile>(QDir(catalogDirectory_).filePath(QStringLiteral("build.lock")));
    writeLock_->setStaleLockTime(30000);
    readOnly_ = !writeLock_->tryLock(0);
    if (readOnly_ && writeLock_->error() != QLockFile::LockFailedError) {
        throw std::runtime_error(
            QStringLiteral("Cannot create catalog lock (error %1): %2")
                .arg(static_cast<int>(writeLock_->error()))
                .arg(writeLock_->fileName())
                .toStdString());
    }

    const QString databasePath = QDir(catalogDirectory_).filePath(QStringLiteral("catalog.sqlite"));
    if (readOnly_ && !QFileInfo::exists(databasePath)) {
        throw std::runtime_error("The catalog is locked and has not been built yet");
    }
    database_ = QSqlDatabase::addDatabase(QStringLiteral("QSQLITE"), connectionName_);
    database_.setDatabaseName(databasePath);
    if (readOnly_) {
        database_.setConnectOptions(QStringLiteral("QSQLITE_OPEN_READONLY"));
    }
    if (!database_.open()) {
        throwSql(QStringLiteral("Cannot open catalog database"), database_.lastError());
    }
    if (!readOnly_) {
        QSqlQuery pragma(database_);
        if (!pragma.exec(QStringLiteral("PRAGMA journal_mode=WAL"))) {
            throwSql(QStringLiteral("Cannot enable SQLite WAL"), pragma.lastError());
        }
        pragma.exec(QStringLiteral("PRAGMA synchronous=NORMAL"));
        pragma.exec(QStringLiteral("PRAGMA foreign_keys=ON"));
        pragma.exec(QStringLiteral("PRAGMA temp_store=MEMORY"));
        createSchema();

        QSqlQuery meta(database_);
        meta.prepare(QStringLiteral("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)"));
        const std::array<std::pair<QString, QString>, 3> values = {{
            {QStringLiteral("source_directory"), sourceDirectory_},
            {QStringLiteral("model_sha256"), modelHash},
            {QStringLiteral("preprocess_version"), QString::number(kPreprocessVersion)},
        }};
        for (const auto& [key, value] : values) {
            meta.bindValue(0, key);
            meta.bindValue(1, value);
            executeOrThrow(meta, QStringLiteral("Cannot write catalog metadata"));
        }
    }
    QSqlQuery featureMeta(database_);
    featureMeta.prepare(QStringLiteral("SELECT value FROM meta WHERE key='feature_file'"));
    executeOrThrow(featureMeta, QStringLiteral("Cannot read feature log metadata"));
    if (featureMeta.next()) {
        featuresPath_ = QDir(catalogDirectory_).filePath(featureMeta.value(0).toString());
    } else if (!readOnly_) {
        QSqlQuery insertFeatureMeta(database_);
        insertFeatureMeta.prepare(QStringLiteral("INSERT INTO meta(key,value) VALUES('feature_file',?)"));
        insertFeatureMeta.addBindValue(QFileInfo(featuresPath_).fileName());
        executeOrThrow(insertFeatureMeta, QStringLiteral("Cannot initialize feature log metadata"));
    }
    refreshActiveState();
}

void CatalogStore::close() {
    if (featureWriteFile_) {
        featureWriteFile_->close();
        featureWriteFile_.reset();
    }
    if (writeBatchActive_) {
        database_.rollback();
        writeBatchActive_ = false;
    }
    activeGenerations_.clear();
    activeImages_.clear();
    if (database_.isValid()) {
        database_.close();
        database_ = QSqlDatabase();
        QSqlDatabase::removeDatabase(connectionName_);
    }
    if (writeLock_) {
        if (!readOnly_) {
            writeLock_->unlock();
        }
        writeLock_.reset();
    }
    sourceDirectory_.clear();
    catalogDirectory_.clear();
    featuresPath_.clear();
    readOnly_ = false;
}

bool CatalogStore::isOpen() const {
    return database_.isOpen();
}

bool CatalogStore::isReadOnly() const {
    return readOnly_;
}

QString CatalogStore::sourceDirectory() const {
    return sourceDirectory_;
}

QString CatalogStore::catalogDirectory() const {
    return catalogDirectory_;
}

void CatalogStore::createSchema() {
    QSqlQuery query(database_);
    const QStringList statements = {
        QStringLiteral("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL)"),
        QStringLiteral(
            "CREATE TABLE IF NOT EXISTS images("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "relative_path TEXT NOT NULL UNIQUE COLLATE NOCASE,"
            "file_size INTEGER NOT NULL DEFAULT -1,"
            "mtime_ms INTEGER NOT NULL DEFAULT -1,"
            "generation INTEGER NOT NULL DEFAULT 0,"
            "active INTEGER NOT NULL DEFAULT 0,"
            "width INTEGER NOT NULL DEFAULT 0,"
            "height INTEGER NOT NULL DEFAULT 0,"
            "has_alpha INTEGER NOT NULL DEFAULT 0,"
            "last_seen INTEGER NOT NULL DEFAULT 0,"
            "error TEXT)"),
        QStringLiteral(
            "CREATE TABLE IF NOT EXISTS vectors("
            "seq INTEGER PRIMARY KEY AUTOINCREMENT,"
            "vector_id INTEGER NOT NULL UNIQUE,"
            "image_id INTEGER NOT NULL,"
            "generation INTEGER NOT NULL,"
            "variant INTEGER NOT NULL,"
            "feature_offset INTEGER NOT NULL,"
            "active INTEGER NOT NULL DEFAULT 1,"
            "FOREIGN KEY(image_id) REFERENCES images(id))"),
        QStringLiteral("CREATE INDEX IF NOT EXISTS vectors_image ON vectors(image_id)"),
        QStringLiteral("CREATE INDEX IF NOT EXISTS vectors_active_seq ON vectors(active,seq)"),
        QStringLiteral("CREATE INDEX IF NOT EXISTS images_seen ON images(last_seen)"),
    };
    for (const QString& statement : statements) {
        if (!query.exec(statement)) {
            throwSql(QStringLiteral("Cannot create catalog schema"), query.lastError());
        }
    }
}

void CatalogStore::ensureWritable() const {
    if (readOnly_) {
        throw std::runtime_error("The catalog is open read-only in another application instance");
    }
}

std::int64_t CatalogStore::beginScan() {
    ensureWritable();
    QSqlQuery query(database_);
    query.prepare(QStringLiteral("SELECT COALESCE(MAX(last_seen),0)+1 FROM images"));
    executeOrThrow(query, QStringLiteral("Cannot allocate scan ID"));
    query.next();
    return query.value(0).toLongLong();
}

FileDecision CatalogStore::inspectFile(
    const QString& relativePath,
    const std::int64_t fileSize,
    const std::int64_t modifiedMilliseconds,
    const std::int64_t scanId) {
    ensureWritable();
    QSqlQuery select(database_);
    select.prepare(QStringLiteral(
        "SELECT id,file_size,mtime_ms,generation,active FROM images WHERE relative_path=?"));
    select.addBindValue(relativePath);
    executeOrThrow(select, QStringLiteral("Cannot inspect catalog image"));

    FileDecision decision;
    decision.relativePath = relativePath;
    if (select.next()) {
        decision.imageId = select.value(0).toLongLong();
        const bool unchanged = select.value(1).toLongLong() == fileSize
            && select.value(2).toLongLong() == modifiedMilliseconds && select.value(4).toBool();
        const int currentGeneration = select.value(3).toInt();
        if (currentGeneration >= 0x3FFF) {
            throw std::runtime_error("Image feature generation limit reached; rebuild the catalog");
        }
        decision.generation = static_cast<std::uint16_t>(currentGeneration + (unchanged ? 0 : 1));
        decision.requiresFeatures = !unchanged;

        QSqlQuery seen(database_);
        seen.prepare(QStringLiteral("UPDATE images SET last_seen=? WHERE id=?"));
        seen.addBindValue(scanId);
        seen.addBindValue(decision.imageId);
        executeOrThrow(seen, QStringLiteral("Cannot update image scan marker"));
    } else {
        QSqlQuery insert(database_);
        insert.prepare(QStringLiteral(
            "INSERT INTO images(relative_path,last_seen) VALUES(?,?)"));
        insert.addBindValue(relativePath);
        insert.addBindValue(scanId);
        executeOrThrow(insert, QStringLiteral("Cannot add catalog image"));
        decision.imageId = insert.lastInsertId().toLongLong();
        decision.generation = 1;
        decision.requiresFeatures = true;
    }
    return decision;
}

void CatalogStore::finishScan(const std::int64_t scanId) {
    ensureWritable();
    if (!database_.transaction()) {
        throwSql(QStringLiteral("Cannot start deletion transaction"), database_.lastError());
    }
    QSqlQuery vectors(database_);
    vectors.prepare(QStringLiteral(
        "UPDATE vectors SET active=0 WHERE image_id IN "
        "(SELECT id FROM images WHERE active=1 AND last_seen<>?)"));
    vectors.addBindValue(scanId);
    if (!vectors.exec()) {
        database_.rollback();
        throwSql(QStringLiteral("Cannot deactivate deleted image vectors"), vectors.lastError());
    }
    QSqlQuery images(database_);
    images.prepare(QStringLiteral("UPDATE images SET active=0 WHERE active=1 AND last_seen<>?"));
    images.addBindValue(scanId);
    if (!images.exec() || !database_.commit()) {
        database_.rollback();
        throwSql(QStringLiteral("Cannot deactivate deleted images"), images.lastError());
    }
    refreshActiveState();
}

void CatalogStore::beginWriteBatch() {
    ensureWritable();
    if (writeBatchActive_ || !database_.transaction()) {
        throwSql(QStringLiteral("Cannot start feature transaction"), database_.lastError());
    }
    writeBatchActive_ = true;
    featureWriteFile_ = std::make_unique<QFile>(featuresPath_);
    if (!featureWriteFile_->open(QIODevice::WriteOnly | QIODevice::Append)) {
        database_.rollback();
        writeBatchActive_ = false;
        featureWriteFile_.reset();
        throw std::runtime_error(("Cannot append feature log: " + featuresPath_).toStdString());
    }
}

std::vector<std::int64_t> CatalogStore::commitImageFeatures(
    const FileDecision& decision,
    const std::int64_t fileSize,
    const std::int64_t modifiedMilliseconds,
    const int width,
    const int height,
    const bool hasTransparency,
    const std::vector<BackgroundVariant>& variants,
    const std::vector<std::vector<float>>& embeddings) {
    ensureWritable();
    if (!writeBatchActive_ || variants.size() != embeddings.size() || variants.empty()) {
        throw std::invalid_argument("Invalid feature batch state");
    }

    QSqlQuery deactivate(database_);
    deactivate.prepare(QStringLiteral("UPDATE vectors SET active=0 WHERE image_id=?"));
    deactivate.addBindValue(decision.imageId);
    executeOrThrow(deactivate, QStringLiteral("Cannot deactivate old image vectors"));

    QSqlQuery insert(database_);
    insert.prepare(QStringLiteral(
        "INSERT INTO vectors(vector_id,image_id,generation,variant,feature_offset,active) "
        "VALUES(?,?,?,?,?,1)"));
    std::vector<std::int64_t> ids;
    ids.reserve(variants.size());
    for (std::size_t index = 0; index < variants.size(); ++index) {
        const std::int64_t vectorId = encodeVectorId(decision.imageId, decision.generation, variants[index]);
        const std::int64_t offset = appendFeature(vectorId, embeddings[index]);
        insert.bindValue(0, vectorId);
        insert.bindValue(1, decision.imageId);
        insert.bindValue(2, decision.generation);
        insert.bindValue(3, static_cast<int>(variants[index]));
        insert.bindValue(4, offset);
        executeOrThrow(insert, QStringLiteral("Cannot store image vector"));
        ids.push_back(vectorId);
    }

    QSqlQuery update(database_);
    update.prepare(QStringLiteral(
        "UPDATE images SET file_size=?,mtime_ms=?,generation=?,active=1,width=?,height=?,"
        "has_alpha=?,error=NULL WHERE id=?"));
    update.addBindValue(fileSize);
    update.addBindValue(modifiedMilliseconds);
    update.addBindValue(decision.generation);
    update.addBindValue(width);
    update.addBindValue(height);
    update.addBindValue(hasTransparency ? 1 : 0);
    update.addBindValue(decision.imageId);
    executeOrThrow(update, QStringLiteral("Cannot activate indexed image"));
    return ids;
}

void CatalogStore::recordImageError(const FileDecision& decision, const QString& error) {
    ensureWritable();
    QSqlQuery deactivate(database_);
    deactivate.prepare(QStringLiteral("UPDATE vectors SET active=0 WHERE image_id=?"));
    deactivate.addBindValue(decision.imageId);
    executeOrThrow(deactivate, QStringLiteral("Cannot deactivate invalid image vectors"));

    QSqlQuery query(database_);
    query.prepare(QStringLiteral("UPDATE images SET generation=?,active=0,error=? WHERE id=?"));
    query.addBindValue(decision.generation);
    query.addBindValue(error.left(2000));
    query.addBindValue(decision.imageId);
    executeOrThrow(query, QStringLiteral("Cannot record image error"));
}

void CatalogStore::commitWriteBatch() {
    if (!writeBatchActive_) {
        return;
    }
    if (!featureWriteFile_ || !featureWriteFile_->flush()) {
        database_.rollback();
        writeBatchActive_ = false;
        featureWriteFile_.reset();
        throw std::runtime_error("Cannot flush feature log batch");
    }
    featureWriteFile_->close();
    featureWriteFile_.reset();
    if (!database_.commit()) {
        database_.rollback();
        writeBatchActive_ = false;
        throwSql(QStringLiteral("Cannot commit feature transaction"), database_.lastError());
    }
    writeBatchActive_ = false;
}

void CatalogStore::rollbackWriteBatch() {
    if (featureWriteFile_) {
        featureWriteFile_->close();
        featureWriteFile_.reset();
    }
    if (writeBatchActive_) {
        database_.rollback();
        writeBatchActive_ = false;
    }
}

std::int64_t CatalogStore::appendFeature(
    const std::int64_t vectorId,
    const std::vector<float>& embedding) {
    if (!writeBatchActive_ || !featureWriteFile_ || !featureWriteFile_->isOpen()) {
        throw std::runtime_error("Feature log is not open for a write batch");
    }
    return writeFeatureRecord(*featureWriteFile_, vectorId, embedding);
}

std::vector<float> CatalogStore::readFeature(const std::int64_t offset) const {
    QFile file(featuresPath_);
    if (!file.open(QIODevice::ReadOnly)) {
        throw std::runtime_error("Cannot open feature log");
    }
    return readFeatureRecord(file, offset);
}

QVector<StoredFeature> CatalogStore::activeFeatureBatch(
    const std::int64_t afterSequence,
    const int limit) const {
    QSqlQuery query(database_);
    query.prepare(QStringLiteral(
        "SELECT seq,vector_id,feature_offset FROM vectors "
        "WHERE active=1 AND seq>? ORDER BY seq LIMIT ?"));
    query.addBindValue(afterSequence);
    query.addBindValue(limit);
    executeOrThrow(query, QStringLiteral("Cannot load feature batch"));
    QVector<StoredFeature> result;
    result.reserve(limit);
    if (!query.next()) {
        return result;
    }
    QFile featureFile(featuresPath_);
    if (!featureFile.open(QIODevice::ReadOnly)) {
        throw std::runtime_error("Cannot open feature log");
    }
    do {
        StoredFeature feature;
        feature.sequence = query.value(0).toLongLong();
        feature.vectorId = query.value(1).toLongLong();
        feature.embedding = readFeatureRecord(featureFile, query.value(2).toLongLong());
        result.push_back(std::move(feature));
    } while (query.next());
    return result;
}

std::int64_t CatalogStore::maxVectorSequence() const {
    QSqlQuery query(database_);
    if (!query.exec(QStringLiteral("SELECT COALESCE(MAX(seq),0) FROM vectors"))) {
        throwSql(QStringLiteral("Cannot read vector sequence"), query.lastError());
    }
    query.next();
    return query.value(0).toLongLong();
}

std::int64_t CatalogStore::activeImageCount() const {
    QSqlQuery query(database_);
    if (!query.exec(QStringLiteral("SELECT COUNT(*) FROM images WHERE active=1"))) {
        throwSql(QStringLiteral("Cannot count active images"), query.lastError());
    }
    query.next();
    return query.value(0).toLongLong();
}

std::int64_t CatalogStore::activeVectorCount() const {
    QSqlQuery query(database_);
    if (!query.exec(QStringLiteral("SELECT COUNT(*) FROM vectors WHERE active=1"))) {
        throwSql(QStringLiteral("Cannot count active vectors"), query.lastError());
    }
    query.next();
    return query.value(0).toLongLong();
}

std::int64_t CatalogStore::totalVectorCount() const {
    QSqlQuery query(database_);
    if (!query.exec(QStringLiteral("SELECT COUNT(*) FROM vectors"))) {
        throwSql(QStringLiteral("Cannot count vectors"), query.lastError());
    }
    query.next();
    return query.value(0).toLongLong();
}

void CatalogStore::refreshActiveState() {
    if (!database_.isOpen()) {
        return;
    }
    QSqlQuery maximum(database_);
    if (!maximum.exec(QStringLiteral("SELECT COALESCE(MAX(id),0) FROM images"))) {
        throwSql(QStringLiteral("Cannot size active image map"), maximum.lastError());
    }
    maximum.next();
    const int maxId = maximum.value(0).toInt();
    activeGenerations_.assign(static_cast<std::size_t>(maxId + 1), 0);
    activeImages_.resize(maxId + 1);
    activeImages_.fill(false);

    QSqlQuery query(database_);
    if (!query.exec(QStringLiteral("SELECT id,generation FROM images WHERE active=1"))) {
        throwSql(QStringLiteral("Cannot load active image map"), query.lastError());
    }
    while (query.next()) {
        const int imageId = query.value(0).toInt();
        activeImages_.setBit(imageId, true);
        activeGenerations_[static_cast<std::size_t>(imageId)] = static_cast<std::uint16_t>(query.value(1).toUInt());
    }
}

bool CatalogStore::isVectorActive(const std::int64_t vectorId) const {
    const std::int64_t imageId = imageIdFromVectorId(vectorId);
    if (imageId <= 0 || imageId >= activeImages_.size() || !activeImages_.testBit(static_cast<int>(imageId))) {
        return false;
    }
    return activeGenerations_[static_cast<std::size_t>(imageId)] == generationFromVectorId(vectorId);
}

QHash<std::int64_t, ImageMetadata> CatalogStore::imageMetadata(
    const QVector<std::int64_t>& imageIds) const {
    QHash<std::int64_t, ImageMetadata> result;
    QSqlQuery query(database_);
    query.prepare(QStringLiteral("SELECT relative_path,width,height,active FROM images WHERE id=?"));
    for (const std::int64_t imageId : imageIds) {
        if (result.contains(imageId)) {
            continue;
        }
        query.bindValue(0, imageId);
        executeOrThrow(query, QStringLiteral("Cannot load result metadata"));
        if (query.next() && query.value(3).toBool()) {
            ImageMetadata metadata;
            metadata.imageId = imageId;
            metadata.relativePath = query.value(0).toString();
            metadata.width = query.value(1).toInt();
            metadata.height = query.value(2).toInt();
            result.insert(imageId, metadata);
        }
        query.finish();
    }
    return result;
}

IndexCheckpoint CatalogStore::checkpoint() const {
    const QString currentPath = QDir(catalogDirectory_).filePath(QStringLiteral("CURRENT.json"));
    QFile file(currentPath);
    if (!file.open(QIODevice::ReadOnly)) {
        return {};
    }
    const QJsonObject object = QJsonDocument::fromJson(file.readAll()).object();
    IndexCheckpoint result;
    result.maxVectorSequence = object.value(QStringLiteral("max_vector_sequence")).toVariant().toLongLong();
    result.path = QDir(catalogDirectory_).filePath(object.value(QStringLiteral("index_file")).toString());
    if (result.maxVectorSequence < 0 || !QFileInfo::exists(result.path)) {
        return {};
    }
    return result;
}

void CatalogStore::commitCheckpoint(
    const QString& indexPath,
    const std::int64_t maxVectorSequence) {
    ensureWritable();
    QJsonObject object;
    object.insert(QStringLiteral("index_file"), QFileInfo(indexPath).fileName());
    object.insert(QStringLiteral("max_vector_sequence"), maxVectorSequence);
    QSaveFile current(QDir(catalogDirectory_).filePath(QStringLiteral("CURRENT.json")));
    if (!current.open(QIODevice::WriteOnly)
        || current.write(QJsonDocument(object).toJson(QJsonDocument::Compact)) < 0
        || !current.commit()) {
        throw std::runtime_error("Cannot commit FAISS checkpoint marker");
    }
}

void CatalogStore::removeObsoleteIndexes(const QString& keepPath) const {
    QDir directory(catalogDirectory_);
    const QFileInfoList files = directory.entryInfoList(
        {QStringLiteral("index_*.faiss")},
        QDir::Files | QDir::NoSymLinks);
    for (const QFileInfo& file : files) {
        if (file.absoluteFilePath() != QFileInfo(keepPath).absoluteFilePath()) {
            QFile::remove(file.absoluteFilePath());
        }
    }
}

void CatalogStore::compactFeatureLog() {
    ensureWritable();
    if (writeBatchActive_) {
        throw std::runtime_error("Cannot compact the feature log during a write batch");
    }
    const QString oldPath = featuresPath_;
    const QString fileName = QStringLiteral("features_%1.fp16").arg(QDateTime::currentMSecsSinceEpoch());
    const QString newPath = QDir(catalogDirectory_).filePath(fileName);
    QSaveFile output(newPath);
    output.setDirectWriteFallback(false);
    if (!output.open(QIODevice::WriteOnly)) {
        throw std::runtime_error("Cannot create compacted feature log");
    }

    QSqlQuery select(database_);
    if (!select.exec(QStringLiteral(
            "SELECT seq,vector_id,feature_offset FROM vectors WHERE active=1 ORDER BY seq"))) {
        throwSql(QStringLiteral("Cannot enumerate active feature records"), select.lastError());
    }
    std::vector<std::pair<std::int64_t, std::int64_t>> offsets;
    offsets.reserve(static_cast<std::size_t>(activeVectorCount()));
    QFile oldInput(oldPath);
    if (!oldInput.open(QIODevice::ReadOnly)) {
        throw std::runtime_error("Cannot open existing feature log for compaction");
    }
    while (select.next()) {
        const std::int64_t sequence = select.value(0).toLongLong();
        const std::int64_t vectorId = select.value(1).toLongLong();
        const std::vector<float> embedding = readFeatureRecord(oldInput, select.value(2).toLongLong());
        offsets.emplace_back(sequence, writeFeatureRecord(output, vectorId, embedding));
    }
    oldInput.close();
    if (!output.commit()) {
        throw std::runtime_error("Cannot commit compacted feature log");
    }

    if (!database_.transaction()) {
        QFile::remove(newPath);
        throwSql(QStringLiteral("Cannot start feature compaction transaction"), database_.lastError());
    }
    QSqlQuery update(database_);
    update.prepare(QStringLiteral("UPDATE vectors SET feature_offset=? WHERE seq=?"));
    for (const auto& [sequence, offset] : offsets) {
        update.bindValue(0, offset);
        update.bindValue(1, sequence);
        if (!update.exec()) {
            database_.rollback();
            QFile::remove(newPath);
            throwSql(QStringLiteral("Cannot update compacted feature offset"), update.lastError());
        }
    }
    QSqlQuery purge(database_);
    if (!purge.exec(QStringLiteral("DELETE FROM vectors WHERE active=0"))) {
        database_.rollback();
        QFile::remove(newPath);
        throwSql(QStringLiteral("Cannot purge inactive vector records"), purge.lastError());
    }
    QSqlQuery meta(database_);
    meta.prepare(QStringLiteral("INSERT OR REPLACE INTO meta(key,value) VALUES('feature_file',?)"));
    meta.addBindValue(fileName);
    if (!meta.exec() || !database_.commit()) {
        database_.rollback();
        QFile::remove(newPath);
        throwSql(QStringLiteral("Cannot activate compacted feature log"), meta.lastError());
    }
    featuresPath_ = newPath;
    if (oldPath != newPath) {
        QFile::remove(oldPath);
    }
}

}  // namespace sf
