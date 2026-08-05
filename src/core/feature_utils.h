#pragma once

#include "core/app_types.h"

#include <QByteArrayView>

#include <cstdint>
#include <functional>
#include <span>
#include <utility>
#include <vector>

namespace sf {

void normalizeL2(std::span<float> values);

std::uint16_t floatToHalf(float value);
float halfToFloat(std::uint16_t value);
std::vector<std::uint16_t> toHalf(std::span<const float> values);
std::vector<float> fromHalf(std::span<const std::uint16_t> values);

std::uint32_t crc32(QByteArrayView bytes);

std::int64_t encodeVectorId(
    std::int64_t imageId,
    std::uint16_t generation,
    BackgroundVariant variant);
std::int64_t imageIdFromVectorId(std::int64_t vectorId);
std::uint16_t generationFromVectorId(std::int64_t vectorId);
BackgroundVariant variantFromVectorId(std::int64_t vectorId);

struct VectorScore {
    std::int64_t vectorId = 0;
    float score = 0.0F;
};

std::vector<std::pair<std::int64_t, float>> rankUniqueImages(
    std::span<const VectorScore> candidates,
    float minSimilarity,
    int topNum,
    const std::function<bool(std::int64_t)>& isActive);

}  // namespace sf
