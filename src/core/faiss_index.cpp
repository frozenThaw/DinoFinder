#include "core/faiss_index.h"

#include <QFile>
#include <QSaveFile>

#include <faiss/IndexHNSW.h>
#include <faiss/IndexIDMap.h>
#include <faiss/impl/io.h>
#include <faiss/index_io.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace sf {

namespace {

class QtReader final : public faiss::IOReader {
public:
    explicit QtReader(QFile& file) : file_(file) {
        name = file.fileName().toStdString();
    }

    std::size_t operator()(void* pointer, const std::size_t size, const std::size_t count) override {
        const qint64 requested = static_cast<qint64>(size * count);
        const qint64 actual = file_.read(static_cast<char*>(pointer), requested);
        if (actual < 0 || (size != 0 && actual % static_cast<qint64>(size) != 0)) {
            throw std::runtime_error("Cannot read FAISS index");
        }
        return size == 0 ? 0 : static_cast<std::size_t>(actual) / size;
    }

private:
    QFile& file_;
};

class QtWriter final : public faiss::IOWriter {
public:
    explicit QtWriter(QSaveFile& file) : file_(file) {
        name = file.fileName().toStdString();
    }

    std::size_t operator()(const void* pointer, const std::size_t size, const std::size_t count) override {
        const qint64 requested = static_cast<qint64>(size * count);
        const qint64 actual = file_.write(static_cast<const char*>(pointer), requested);
        if (actual != requested) {
            throw std::runtime_error("Cannot write FAISS index");
        }
        return count;
    }

private:
    QSaveFile& file_;
};

faiss::IndexHNSW* hnswIndex(faiss::IndexIDMap2* index) {
    return dynamic_cast<faiss::IndexHNSW*>(index->index);
}

const faiss::IndexHNSW* hnswIndex(const faiss::IndexIDMap2* index) {
    return dynamic_cast<const faiss::IndexHNSW*>(index->index);
}

}  // namespace

FaissIndex::FaissIndex() = default;
FaissIndex::~FaissIndex() = default;

void FaissIndex::create() {
    std::scoped_lock lock(mutex_);
    auto* hnsw = new faiss::IndexHNSWFlat(
        kEmbeddingDimension,
        kHnswLinks,
        faiss::METRIC_INNER_PRODUCT);
    hnsw->hnsw.efConstruction = kHnswEfConstruction;
    auto idMapIndex = std::make_unique<faiss::IndexIDMap2>(hnsw);
    idMapIndex->own_fields = true;
    index_ = std::move(idMapIndex);
}

void FaissIndex::clear() {
    std::scoped_lock lock(mutex_);
    index_.reset();
}

void FaissIndex::load(const QString& path) {
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        throw std::runtime_error(("Cannot open FAISS index: " + path).toStdString());
    }
    QtReader reader(file);
    std::unique_ptr<faiss::Index> loaded(faiss::read_index(&reader));
    auto* mapped = dynamic_cast<faiss::IndexIDMap2*>(loaded.get());
    if (mapped == nullptr || mapped->d != kEmbeddingDimension || hnswIndex(mapped) == nullptr) {
        throw std::runtime_error("The FAISS index has an incompatible type or dimension");
    }
    std::scoped_lock lock(mutex_);
    index_ = std::move(loaded);
}

void FaissIndex::saveAtomic(const QString& path) const {
    std::scoped_lock lock(mutex_);
    if (!index_) {
        throw std::runtime_error("Cannot save an empty FAISS index");
    }
    QSaveFile file(path);
    file.setDirectWriteFallback(false);
    if (!file.open(QIODevice::WriteOnly)) {
        throw std::runtime_error(("Cannot create FAISS index: " + path).toStdString());
    }
    QtWriter writer(file);
    faiss::write_index(index_.get(), &writer);
    if (!file.commit()) {
        throw std::runtime_error(("Cannot commit FAISS index: " + path).toStdString());
    }
}

void FaissIndex::add(
    const std::span<const std::int64_t> ids,
    const std::span<const std::vector<float>> embeddings) {
    if (ids.size() != embeddings.size()) {
        throw std::invalid_argument("FAISS IDs and embeddings have different lengths");
    }
    if (ids.empty()) {
        return;
    }
    std::vector<float> contiguous;
    contiguous.reserve(ids.size() * kEmbeddingDimension);
    for (const auto& embedding : embeddings) {
        if (embedding.size() != kEmbeddingDimension) {
            throw std::invalid_argument("FAISS embedding has an invalid dimension");
        }
        contiguous.insert(contiguous.end(), embedding.begin(), embedding.end());
    }
    std::scoped_lock lock(mutex_);
    if (!index_) {
        throw std::runtime_error("FAISS index is not initialized");
    }
    idMap()->add_with_ids(
        static_cast<faiss::idx_t>(ids.size()),
        contiguous.data(),
        reinterpret_cast<const faiss::idx_t*>(ids.data()));
}

std::vector<std::pair<std::int64_t, float>> FaissIndex::search(
    const std::span<const float> normalizedEmbedding,
    const int count,
    const int efSearch) const {
    if (normalizedEmbedding.size() != kEmbeddingDimension || count <= 0) {
        throw std::invalid_argument("Invalid FAISS search parameters");
    }
    std::scoped_lock lock(mutex_);
    if (!index_ || index_->ntotal == 0) {
        return {};
    }
    auto* hnsw = const_cast<faiss::IndexHNSW*>(hnswIndex(idMap()));
    hnsw->hnsw.efSearch = std::max(count, efSearch);
    const int actualCount = std::min<std::int64_t>(count, index_->ntotal);
    std::vector<float> distances(static_cast<std::size_t>(actualCount));
    std::vector<faiss::idx_t> labels(static_cast<std::size_t>(actualCount));
    index_->search(1, normalizedEmbedding.data(), actualCount, distances.data(), labels.data());

    std::vector<std::pair<std::int64_t, float>> result;
    result.reserve(actualCount);
    for (int position = 0; position < actualCount; ++position) {
        if (labels[position] >= 0) {
            result.emplace_back(static_cast<std::int64_t>(labels[position]), distances[position]);
        }
    }
    return result;
}

std::int64_t FaissIndex::size() const {
    std::scoped_lock lock(mutex_);
    return index_ ? index_->ntotal : 0;
}

bool FaissIndex::empty() const {
    return size() == 0;
}

faiss::IndexIDMap2* FaissIndex::idMap() {
    auto* result = dynamic_cast<faiss::IndexIDMap2*>(index_.get());
    if (result == nullptr) {
        throw std::runtime_error("FAISS index is not an IndexIDMap2");
    }
    return result;
}

const faiss::IndexIDMap2* FaissIndex::idMap() const {
    auto* result = dynamic_cast<const faiss::IndexIDMap2*>(index_.get());
    if (result == nullptr) {
        throw std::runtime_error("FAISS index is not an IndexIDMap2");
    }
    return result;
}

}  // namespace sf
