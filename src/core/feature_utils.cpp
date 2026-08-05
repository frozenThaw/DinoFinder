#include "core/feature_utils.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <unordered_map>

namespace sf {

namespace {

constexpr std::uint64_t kVariantMask = 0x3ULL;
constexpr std::uint64_t kGenerationMask = 0x3FFFULL;
constexpr int kGenerationShift = 2;
constexpr int kImageShift = 16;

}  // namespace

void normalizeL2(const std::span<float> values) {
    double squaredNorm = 0.0;
    for (const float value : values) {
        squaredNorm += static_cast<double>(value) * static_cast<double>(value);
    }
    if (squaredNorm <= std::numeric_limits<double>::epsilon()) {
        throw std::runtime_error("Cannot normalize a zero-length embedding");
    }
    const float inverseNorm = static_cast<float>(1.0 / std::sqrt(squaredNorm));
    for (float& value : values) {
        value *= inverseNorm;
    }
}

std::uint16_t floatToHalf(const float value) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t sign = (bits >> 16U) & 0x8000U;
    std::int32_t exponent = static_cast<std::int32_t>((bits >> 23U) & 0xFFU) - 127 + 15;
    std::uint32_t mantissa = bits & 0x7FFFFFU;

    if (exponent <= 0) {
        if (exponent < -10) {
            return static_cast<std::uint16_t>(sign);
        }
        mantissa = (mantissa | 0x800000U) >> static_cast<std::uint32_t>(1 - exponent);
        return static_cast<std::uint16_t>(sign | ((mantissa + 0x1000U) >> 13U));
    }
    if (exponent >= 31) {
        return static_cast<std::uint16_t>(sign | 0x7C00U | (mantissa != 0U ? 0x0200U : 0U));
    }
    return static_cast<std::uint16_t>(
        sign | (static_cast<std::uint32_t>(exponent) << 10U) | ((mantissa + 0x1000U) >> 13U));
}

float halfToFloat(const std::uint16_t value) {
    const std::uint32_t sign = (static_cast<std::uint32_t>(value & 0x8000U)) << 16U;
    std::uint32_t exponent = (value >> 10U) & 0x1FU;
    std::uint32_t mantissa = value & 0x03FFU;
    std::uint32_t bits = 0;

    if (exponent == 0U) {
        if (mantissa == 0U) {
            bits = sign;
        } else {
            int unbiasedExponent = -14;
            while ((mantissa & 0x0400U) == 0U) {
                mantissa <<= 1U;
                --unbiasedExponent;
            }
            mantissa &= 0x03FFU;
            bits = sign
                | (static_cast<std::uint32_t>(unbiasedExponent + 127) << 23U)
                | (mantissa << 13U);
        }
    } else if (exponent == 31U) {
        bits = sign | 0x7F800000U | (mantissa << 13U);
    } else {
        bits = sign | ((exponent + 112U) << 23U) | (mantissa << 13U);
    }
    return std::bit_cast<float>(bits);
}

std::vector<std::uint16_t> toHalf(const std::span<const float> values) {
    std::vector<std::uint16_t> result;
    result.reserve(values.size());
    std::transform(values.begin(), values.end(), std::back_inserter(result), floatToHalf);
    return result;
}

std::vector<float> fromHalf(const std::span<const std::uint16_t> values) {
    std::vector<float> result;
    result.reserve(values.size());
    std::transform(values.begin(), values.end(), std::back_inserter(result), halfToFloat);
    return result;
}

std::uint32_t crc32(const QByteArrayView bytes) {
    std::uint32_t crc = 0xFFFFFFFFU;
    for (const char byte : bytes) {
        crc ^= static_cast<std::uint8_t>(byte);
        for (int bit = 0; bit < 8; ++bit) {
            crc = (crc >> 1U) ^ ((crc & 1U) != 0U ? 0xEDB88320U : 0U);
        }
    }
    return crc ^ 0xFFFFFFFFU;
}

std::int64_t encodeVectorId(
    const std::int64_t imageId,
    const std::uint16_t generation,
    const BackgroundVariant variant) {
    if (imageId <= 0 || imageId >= (1LL << 47)) {
        throw std::out_of_range("Image ID exceeds the vector ID encoding range");
    }
    if (generation == 0 || generation > kGenerationMask) {
        throw std::out_of_range("Feature generation exceeds the vector ID encoding range");
    }
    const auto encoded = (static_cast<std::uint64_t>(imageId) << kImageShift)
        | (static_cast<std::uint64_t>(generation) << kGenerationShift)
        | static_cast<std::uint8_t>(variant);
    return static_cast<std::int64_t>(encoded);
}

std::int64_t imageIdFromVectorId(const std::int64_t vectorId) {
    return static_cast<std::int64_t>(static_cast<std::uint64_t>(vectorId) >> kImageShift);
}

std::uint16_t generationFromVectorId(const std::int64_t vectorId) {
    return static_cast<std::uint16_t>(
        (static_cast<std::uint64_t>(vectorId) >> kGenerationShift) & kGenerationMask);
}

BackgroundVariant variantFromVectorId(const std::int64_t vectorId) {
    return static_cast<BackgroundVariant>(static_cast<std::uint64_t>(vectorId) & kVariantMask);
}

std::vector<std::pair<std::int64_t, float>> rankUniqueImages(
    const std::span<const VectorScore> candidates,
    const float minSimilarity,
    const int topNum,
    const std::function<bool(std::int64_t)>& isActive) {
    if (topNum <= 0) {
        return {};
    }
    std::unordered_map<std::int64_t, float> bestByImage;
    bestByImage.reserve(candidates.size());
    for (const VectorScore& candidate : candidates) {
        if (candidate.score < minSimilarity || !isActive(candidate.vectorId)) {
            continue;
        }
        const std::int64_t imageId = imageIdFromVectorId(candidate.vectorId);
        const auto [position, inserted] = bestByImage.emplace(imageId, candidate.score);
        if (!inserted && candidate.score > position->second) {
            position->second = candidate.score;
        }
    }
    std::vector<std::pair<std::int64_t, float>> ranked(bestByImage.begin(), bestByImage.end());
    std::ranges::sort(ranked, [](const auto& left, const auto& right) {
        return left.second == right.second ? left.first < right.first : left.second > right.second;
    });
    if (ranked.size() > static_cast<std::size_t>(topNum)) {
        ranked.resize(static_cast<std::size_t>(topNum));
    }
    return ranked;
}

}  // namespace sf
