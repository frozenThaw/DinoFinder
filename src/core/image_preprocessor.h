#pragma once

#include "core/app_types.h"

#include <QString>

namespace sf {

class ImagePreprocessor final {
public:
    [[nodiscard]] PreparedImage prepare(const QString& path) const;
};

}  // namespace sf
