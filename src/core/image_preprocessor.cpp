#include "core/image_preprocessor.h"

#include <QFile>
#include <QBuffer>
#include <QImageIOHandler>
#include <QImageReader>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <array>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace sf {

namespace {

void applyExifTransformation(cv::Mat& image, const QImageIOHandler::Transformations transformation) {
    if (transformation.testFlag(QImageIOHandler::TransformationRotate90)) {
        cv::rotate(image, image, cv::ROTATE_90_CLOCKWISE);
    }
    const bool mirror = transformation.testFlag(QImageIOHandler::TransformationMirror);
    const bool flip = transformation.testFlag(QImageIOHandler::TransformationFlip);
    if (mirror && flip) {
        cv::rotate(image, image, cv::ROTATE_180);
    } else if (mirror) {
        cv::flip(image, image, 1);
    } else if (flip) {
        cv::flip(image, image, 0);
    }
}

cv::Mat decodeImage(const QString& path) {
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        throw std::runtime_error(("Cannot open image: " + path).toStdString());
    }
    const QByteArray bytes = file.readAll();
    QBuffer metadataBuffer;
    metadataBuffer.setData(bytes);
    metadataBuffer.open(QIODevice::ReadOnly);
    QImageReader metadataReader(&metadataBuffer);
    metadataReader.setDecideFormatFromContent(true);
    const QImageIOHandler::Transformations transformation = metadataReader.transformation();
    const std::vector<std::uint8_t> buffer(bytes.begin(), bytes.end());
    cv::Mat image = cv::imdecode(buffer, cv::IMREAD_UNCHANGED);
    if (image.empty()) {
        throw std::runtime_error(("OpenCV cannot decode image: " + path).toStdString());
    }
    if (image.depth() == CV_16U) {
        image.convertTo(image, CV_MAKETYPE(CV_8U, image.channels()), 1.0 / 257.0);
    } else if (image.depth() != CV_8U) {
        double minimum = 0.0;
        double maximum = 0.0;
        cv::minMaxLoc(image.reshape(1), &minimum, &maximum);
        const double scale = maximum > minimum ? 255.0 / (maximum - minimum) : 1.0;
        image.convertTo(image, CV_MAKETYPE(CV_8U, image.channels()), scale, -minimum * scale);
    }
    applyExifTransformation(image, transformation);
    return image;
}

cv::Mat toBgra(const cv::Mat& source) {
    cv::Mat result;
    switch (source.channels()) {
    case 1:
        cv::cvtColor(source, result, cv::COLOR_GRAY2BGRA);
        break;
    case 3:
        cv::cvtColor(source, result, cv::COLOR_BGR2BGRA);
        break;
    case 4:
        result = source.clone();
        break;
    default:
        throw std::runtime_error("Unsupported image channel count");
    }
    return result;
}

bool hasTransparency(const cv::Mat& bgra) {
    std::vector<cv::Mat> channels;
    cv::split(bgra, channels);
    double minimum = 255.0;
    cv::minMaxLoc(channels[3], &minimum, nullptr);
    return minimum < 255.0;
}

cv::Mat composite(const cv::Mat& bgra, const cv::Vec3b background) {
    cv::Mat result(bgra.rows, bgra.cols, CV_8UC3);
    for (int row = 0; row < bgra.rows; ++row) {
        const auto* source = bgra.ptr<cv::Vec4b>(row);
        auto* destination = result.ptr<cv::Vec3b>(row);
        for (int column = 0; column < bgra.cols; ++column) {
            const int alpha = source[column][3];
            for (int channel = 0; channel < 3; ++channel) {
                destination[column][channel] = static_cast<std::uint8_t>(
                    (source[column][channel] * alpha + background[channel] * (255 - alpha) + 127)
                    / 255);
            }
        }
    }
    return result;
}

cv::Vec3b cornerColor(const cv::Mat& bgr) {
    const std::array<cv::Vec3b, 4> corners = {
        bgr.at<cv::Vec3b>(0, 0),
        bgr.at<cv::Vec3b>(0, bgr.cols - 1),
        bgr.at<cv::Vec3b>(bgr.rows - 1, 0),
        bgr.at<cv::Vec3b>(bgr.rows - 1, bgr.cols - 1),
    };
    cv::Vec3b color;
    for (int channel = 0; channel < 3; ++channel) {
        int sum = 0;
        for (const auto& corner : corners) {
            sum += corner[channel];
        }
        const int quotient = sum / 4;
        const int remainder = sum % 4;
        const int rounded = remainder < 2 ? quotient
            : remainder > 2               ? quotient + 1
                                          : quotient + (quotient & 1);
        color[channel] = static_cast<std::uint8_t>(rounded);
    }
    return color;
}

cv::Mat squarePad(const cv::Mat& bgr, const cv::Vec3b color) {
    const int size = std::max(bgr.rows, bgr.cols);
    const int top = (size - bgr.rows) / 2;
    const int bottom = size - bgr.rows - top;
    const int left = (size - bgr.cols) / 2;
    const int right = size - bgr.cols - left;
    cv::Mat result;
    cv::copyMakeBorder(
        bgr,
        result,
        top,
        bottom,
        left,
        right,
        cv::BORDER_CONSTANT,
        cv::Scalar(color[0], color[1], color[2]));
    return result;
}

ImageTensor makeTensor(const cv::Mat& bgr, const cv::Vec3b padding, const BackgroundVariant variant) {
    cv::Mat resized;
    cv::resize(squarePad(bgr, padding), resized, cv::Size(kImageSize, kImageSize), 0.0, 0.0, cv::INTER_CUBIC);
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);

    constexpr std::array<float, 3> mean = {0.485F, 0.456F, 0.406F};
    constexpr std::array<float, 3> standardDeviation = {0.229F, 0.224F, 0.225F};
    std::array<cv::Mat, 3> channels;
    cv::split(resized, channels.data());

    ImageTensor tensor;
    tensor.variant = variant;
    tensor.nchw.resize(3ULL * kImageSize * kImageSize);
    const std::size_t planeSize = static_cast<std::size_t>(kImageSize) * kImageSize;
    for (int channel = 0; channel < 3; ++channel) {
        channels[channel] = (channels[channel] - mean[channel]) / standardDeviation[channel];
        if (!channels[channel].isContinuous()) {
            channels[channel] = channels[channel].clone();
        }
        std::memcpy(
            tensor.nchw.data() + static_cast<std::size_t>(channel) * planeSize,
            channels[channel].ptr<float>(),
            planeSize * sizeof(float));
    }
    return tensor;
}

}  // namespace

PreparedImage ImagePreprocessor::prepare(const QString& path) const {
    const cv::Mat source = decodeImage(path);
    const cv::Mat bgra = toBgra(source);

    PreparedImage result;
    result.width = source.cols;
    result.height = source.rows;
    result.hasTransparency = hasTransparency(bgra);

    if (result.hasTransparency) {
        const std::array<std::pair<BackgroundVariant, cv::Vec3b>, 3> backgrounds = {{
            {BackgroundVariant::White, cv::Vec3b(255, 255, 255)},
            {BackgroundVariant::Gray, cv::Vec3b(127, 127, 127)},
            {BackgroundVariant::Black, cv::Vec3b(0, 0, 0)},
        }};
        for (const auto& [variant, color] : backgrounds) {
            result.tensors.push_back(makeTensor(composite(bgra, color), color, variant));
        }
    } else {
        cv::Mat bgr;
        cv::cvtColor(bgra, bgr, cv::COLOR_BGRA2BGR);
        const cv::Vec3b padding = cornerColor(bgr);
        result.tensors.push_back(makeTensor(bgr, padding, BackgroundVariant::Original));
    }
    return result;
}

}  // namespace sf
