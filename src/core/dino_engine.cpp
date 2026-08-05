#include "core/dino_engine.h"

#include "core/feature_utils.h"

#include <QCryptographicHash>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace sf {

struct DinoEngine::Impl {
    Ort::Env environment{ORT_LOGGING_LEVEL_WARNING, "SimilarFinder"};
    std::unique_ptr<Ort::Session> session;
    QString modelPath;
    QString inputName = QStringLiteral("pixel_values");
    QString outputName = QStringLiteral("cls_embedding");
    bool cuda = false;
    int currentBatchSize = 8;

    bool verifyModel(const QString& manifestPath, QString* errorMessage) {
        QFile manifestFile(manifestPath);
        if (!manifestFile.open(QIODevice::ReadOnly)) {
            *errorMessage = QStringLiteral("Cannot open model manifest: %1").arg(manifestPath);
            return false;
        }
        const QJsonDocument document = QJsonDocument::fromJson(manifestFile.readAll());
        const QJsonObject manifest = document.object();
        const QString expectedHash = manifest.value(QStringLiteral("sha256")).toString().toLower();
        inputName = manifest.value(QStringLiteral("input_name")).toString(QStringLiteral("pixel_values"));
        outputName = manifest.value(QStringLiteral("output_name")).toString(QStringLiteral("cls_embedding"));
        if (manifest.value(QStringLiteral("dimension")).toInt() != kEmbeddingDimension
            || manifest.value(QStringLiteral("image_size")).toInt() != kImageSize) {
            *errorMessage = QStringLiteral("The model manifest is incompatible with this application");
            return false;
        }
        if (expectedHash.size() != 64) {
            *errorMessage = QStringLiteral("The model manifest does not contain a valid SHA-256 hash");
            return false;
        }

        QFile modelFile(modelPath);
        if (!modelFile.open(QIODevice::ReadOnly)) {
            *errorMessage = QStringLiteral("Cannot open DINOv2 model: %1").arg(modelPath);
            return false;
        }
        QCryptographicHash hasher(QCryptographicHash::Sha256);
        if (!hasher.addData(&modelFile)) {
            *errorMessage = QStringLiteral("Cannot hash DINOv2 model: %1").arg(modelPath);
            return false;
        }
        const QString actualHash = QString::fromLatin1(hasher.result().toHex());
        if (actualHash != expectedHash) {
            *errorMessage = QStringLiteral("DINOv2 model checksum mismatch");
            return false;
        }
        return true;
    }

    void createSession(const bool useCuda) {
        Ort::SessionOptions options;
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        options.DisableMemPattern();
#ifndef SIMILARFINDER_CPU_ONLY
        if (useCuda) {
            OrtCUDAProviderOptions cudaOptions{};
            cudaOptions.device_id = 0;
            cudaOptions.arena_extend_strategy = 0;
            cudaOptions.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
            cudaOptions.do_copy_in_default_stream = 1;
            options.AppendExecutionProvider_CUDA(cudaOptions);
        }
#else
        if (useCuda) {
            throw std::logic_error("CUDA is disabled in this build");
        }
#endif
#ifdef _WIN32
        session = std::make_unique<Ort::Session>(environment, modelPath.toStdWString().c_str(), options);
#else
        session = std::make_unique<Ort::Session>(environment, modelPath.toUtf8().constData(), options);
#endif
        cuda = useCuda;
        currentBatchSize = cuda ? 128 : 8;
    }

    std::vector<std::vector<float>> run(
        const std::span<const ImageTensor> tensors,
        const int requestedBatchSize) const {
        if (!session) {
            throw std::runtime_error("DINOv2 session is not initialized");
        }
        constexpr std::size_t tensorSize = 3ULL * kImageSize * kImageSize;
        std::vector<std::vector<float>> embeddings;
        embeddings.reserve(tensors.size());
        const std::string inputNameUtf8 = inputName.toStdString();
        const std::string outputNameUtf8 = outputName.toStdString();
        const std::array<const char*, 1> inputNames = {inputNameUtf8.c_str()};
        const std::array<const char*, 1> outputNames = {outputNameUtf8.c_str()};
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        for (std::size_t start = 0; start < tensors.size(); start += requestedBatchSize) {
            const std::size_t count = std::min<std::size_t>(requestedBatchSize, tensors.size() - start);
            std::vector<float> batch(count * tensorSize);
            for (std::size_t item = 0; item < count; ++item) {
                if (tensors[start + item].nchw.size() != tensorSize) {
                    throw std::runtime_error("DINOv2 input tensor has an invalid size");
                }
                std::memcpy(
                    batch.data() + item * tensorSize,
                    tensors[start + item].nchw.data(),
                    tensorSize * sizeof(float));
            }
            const std::array<std::int64_t, 4> shape = {
                static_cast<std::int64_t>(count), 3, kImageSize, kImageSize};
            Ort::Value input = Ort::Value::CreateTensor<float>(
                memoryInfo,
                batch.data(),
                batch.size(),
                shape.data(),
                shape.size());
            auto outputs = session->Run(
                Ort::RunOptions{nullptr},
                inputNames.data(),
                &input,
                1,
                outputNames.data(),
                1);
            if (outputs.size() != 1 || !outputs[0].IsTensor()) {
                throw std::runtime_error("DINOv2 did not return a tensor output");
            }
            const auto outputShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
            if (outputShape.size() != 2 || outputShape[0] != static_cast<std::int64_t>(count)
                || outputShape[1] != kEmbeddingDimension) {
                throw std::runtime_error("DINOv2 output must have shape [N, 384]");
            }
            const float* output = outputs[0].GetTensorData<float>();
            for (std::size_t item = 0; item < count; ++item) {
                std::vector<float> embedding(
                    output + item * kEmbeddingDimension,
                    output + (item + 1) * kEmbeddingDimension);
                normalizeL2(embedding);
                embeddings.push_back(std::move(embedding));
            }
        }
        return embeddings;
    }
};

DinoEngine::DinoEngine() : impl_(std::make_unique<Impl>()) {}
DinoEngine::~DinoEngine() = default;

bool DinoEngine::initialize(
    const QString& modelPath,
    const QString& manifestPath,
    const bool preferCuda,
    QString* errorMessage) {
    QString localError;
    if (errorMessage == nullptr) {
        errorMessage = &localError;
    }
    impl_->modelPath = modelPath;
    if (!impl_->verifyModel(manifestPath, errorMessage)) {
        return false;
    }
    bool requestCuda = preferCuda;
#ifdef SIMILARFINDER_CPU_ONLY
    requestCuda = false;
#endif
    try {
        impl_->createSession(requestCuda);
    } catch (const std::exception& cudaError) {
        if (!requestCuda) {
            *errorMessage = QString::fromUtf8(cudaError.what());
            return false;
        }
        try {
            impl_->createSession(false);
        } catch (const std::exception& cpuError) {
            *errorMessage = QStringLiteral("CUDA failed: %1\nCPU fallback failed: %2")
                                .arg(QString::fromUtf8(cudaError.what()), QString::fromUtf8(cpuError.what()));
            return false;
        }
    }
    return true;
}

std::vector<std::vector<float>> DinoEngine::embed(const std::span<const ImageTensor> tensors) {
    if (tensors.empty()) {
        return {};
    }
    int attemptBatchSize = impl_->currentBatchSize;
    while (true) {
        try {
            auto result = impl_->run(tensors, attemptBatchSize);
            impl_->currentBatchSize = attemptBatchSize;
            return result;
        } catch (const Ort::Exception&) {
            if (impl_->cuda && attemptBatchSize > 16) {
                attemptBatchSize = std::max(16, attemptBatchSize / 2);
                continue;
            }
            if (impl_->cuda) {
                impl_->createSession(false);
                attemptBatchSize = 8;
                continue;
            }
            throw;
        }
    }
}

bool DinoEngine::isReady() const {
    return impl_->session != nullptr;
}

bool DinoEngine::isCuda() const {
    return impl_->cuda;
}

QString DinoEngine::providerName() const {
    return impl_->cuda ? QStringLiteral("CUDA") : QStringLiteral("CPU");
}

int DinoEngine::batchSize() const {
    return impl_->currentBatchSize;
}

}  // namespace sf
