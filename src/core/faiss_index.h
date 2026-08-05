#pragma once

#include "core/app_types.h"

#include <QString>

#include <faiss/IndexIDMap.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <utility>
#include <vector>

namespace sf {

class FaissIndex final {
public:
    FaissIndex();
    ~FaissIndex();

    FaissIndex(const FaissIndex&) = delete;
    FaissIndex& operator=(const FaissIndex&) = delete;

    void create();
    void clear();
    void load(const QString& path);
    void saveAtomic(const QString& path) const;

    void add(
        std::span<const std::int64_t> ids,
        std::span<const std::vector<float>> embeddings);

    [[nodiscard]] std::vector<std::pair<std::int64_t, float>> search(
        std::span<const float> normalizedEmbedding,
        int count,
        int efSearch) const;

    [[nodiscard]] std::int64_t size() const;
    [[nodiscard]] bool empty() const;

private:
    faiss::IndexIDMap2* idMap();
    const faiss::IndexIDMap2* idMap() const;

    mutable std::mutex mutex_;
    std::unique_ptr<faiss::Index> index_;
};

}  // namespace sf
