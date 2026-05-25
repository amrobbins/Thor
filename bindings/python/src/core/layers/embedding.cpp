#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>

#include <memory>
#include <optional>

#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Api/Layers/Learning/Embedding.h"
#include "DeepLearning/Api/Layers/Learning/TrainableLayer.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

using DataType = ThorImplementation::DataType;

namespace {
std::optional<DataType> optionalDataTypeFromPython(const nb::object& obj) {
    if (obj.is_none())
        return std::nullopt;
    return nb::cast<DataType>(obj);
}

std::optional<uint64_t> optionalUInt64FromPython(const nb::object& obj) {
    if (obj.is_none())
        return std::nullopt;
    return nb::cast<uint64_t>(obj);
}
}  // namespace

void bind_embedding(nb::module_& m) {
    auto embedding = nb::class_<Embedding, TrainableLayer>(m, "Embedding");
    embedding.attr("__module__") = "thor.layers";

    embedding.def(
        "__init__",
        [](Embedding* self,
           Network& network,
           Tensor featureInput,
           uint64_t vocabularySize,
           uint64_t embeddingDim,
           nb::object weightsDataTypeObj,
           nb::object paddingIndexObj,
           bool sparseGradients,
           shared_ptr<Initializer> weightsInitializer,
           shared_ptr<Optimizer> weightsOptimizer) {
            Embedding::Builder builder;
            builder.network(network)
                .featureInput(featureInput)
                .vocabularySize(vocabularySize)
                .embeddingDim(embeddingDim)
                .sparseGradients(sparseGradients);

            std::optional<DataType> weightsDataType = optionalDataTypeFromPython(weightsDataTypeObj);
            if (weightsDataType.has_value())
                builder.weightsDataType(weightsDataType.value());

            std::optional<uint64_t> paddingIndex = optionalUInt64FromPython(paddingIndexObj);
            if (paddingIndex.has_value())
                builder.paddingIndex(paddingIndex.value());

            if (weightsInitializer != nullptr)
                builder.weightsInitializer(weightsInitializer);
            if (weightsOptimizer != nullptr)
                builder.weightsOptimizer(weightsOptimizer);

            Embedding built = builder.build();
            new (self) Embedding(std::move(built));
        },
        "network"_a,
        "feature_input"_a,
        "vocabulary_size"_a,
        "embedding_dim"_a,
        "weights_data_type"_a.none() = nb::none(),
        "padding_index"_a.none() = nb::none(),
        "sparse_gradients"_a = true,
        "weights_initializer"_a.none() = nb::none(),
        "weights_optimizer"_a.none() = nb::none());

    embedding.def(
        "get_feature_output",
        [](Embedding& self) -> Tensor {
            std::optional<Tensor> maybeFeatureOutput = self.getFeatureOutput();
            return maybeFeatureOutput.value();
        },
        R"nbdoc(
            Return the output tensor produced by this layer.
            )nbdoc");

    embedding.def_prop_ro("vocabulary_size", &Embedding::getVocabularySize);
    embedding.def_prop_ro("embedding_dim", &Embedding::getEmbeddingDim);
    embedding.def_prop_ro("weights_data_type", &Embedding::getWeightsDataType);
    embedding.def_prop_ro("padding_index", &Embedding::getPaddingIndex);
    embedding.def_prop_ro("sparse_gradients", &Embedding::usesSparseGradients);

    embedding.attr("__doc__") = R"nbdoc(
        Sparse-gradient embedding lookup layer.

        Embedding maps an integer tensor of token ids to a floating output tensor whose
        final dimension is ``embedding_dim``. Its output shape is the input index shape
        with ``embedding_dim`` appended.

        This layer intentionally does not implement dense table gradients. Training uses
        sparse row updates; the first backend slice supports plain SGD for fp32 embedding
        tables, with fp16/bf16/fp32 forward lookup support.
        )nbdoc";
}
