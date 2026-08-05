#pragma once

#include "core/app_types.h"

#include <QString>

#include <memory>
#include <span>
#include <vector>

namespace sf {

class DinoEngine final {
public:
    DinoEngine();
    ~DinoEngine();

    DinoEngine(const DinoEngine&) = delete;
    DinoEngine& operator=(const DinoEngine&) = delete;

    bool initialize(
        const QString& modelPath,
        const QString& manifestPath,
        bool preferCuda,
        QString* errorMessage);

    [[nodiscard]] std::vector<std::vector<float>> embed(
        std::span<const ImageTensor> tensors);

    [[nodiscard]] bool isReady() const;
    [[nodiscard]] bool isCuda() const;
    [[nodiscard]] QString providerName() const;
    [[nodiscard]] int batchSize() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace sf
