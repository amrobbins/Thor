#include "DeepLearning/Api/Data/DatasetSplitManifest.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

namespace Thor {
namespace {

using nlohmann::json;
constexpr const char *FORMAT = "thor.dataset_split_manifest.v1";

uint64_t checkedAdd(uint64_t left, uint64_t right, const char *context) {
    if (left > std::numeric_limits<uint64_t>::max() - right) {
        throw std::runtime_error(std::string(context) + " overflow.");
    }
    return left + right;
}

uint64_t checkedMul(uint64_t left, uint64_t right, const char *context) {
    if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left) {
        throw std::runtime_error(std::string(context) + " overflow.");
    }
    return left * right;
}

__int128 extendedGcdCoefficient(uint64_t a, uint64_t b) {
    __int128 oldR = static_cast<__int128>(a);
    __int128 r = static_cast<__int128>(b);
    __int128 oldS = 1;
    __int128 s = 0;
    while (r != 0) {
        const __int128 quotient = oldR / r;
        const __int128 nextR = oldR - quotient * r;
        oldR = r;
        r = nextR;
        const __int128 nextS = oldS - quotient * s;
        oldS = s;
        s = nextS;
    }
    return oldS;
}

uint64_t normalizeModulo(__int128 value, uint64_t modulus) {
    if (modulus == 1) {
        return 0;
    }
    const __int128 signedModulus = static_cast<__int128>(modulus);
    __int128 normalized = value % signedModulus;
    if (normalized < 0) {
        normalized += signedModulus;
    }
    return static_cast<uint64_t>(normalized);
}

bool rangesIntersect(const ExampleIndexRange &left, const ExampleIndexRange &right) {
    const uint64_t lower = std::max(left.start, right.start);
    const uint64_t upper = std::min(left.last(), right.last());
    if (lower > upper) {
        return false;
    }

    const uint64_t gcd = std::gcd(left.stride, right.stride);
    const __int128 difference = static_cast<__int128>(right.start) - static_cast<__int128>(left.start);
    if (difference % static_cast<__int128>(gcd) != 0) {
        return false;
    }

    const uint64_t leftReduced = left.stride / gcd;
    const uint64_t rightReduced = right.stride / gcd;
    uint64_t leftStepCount = 0;
    if (rightReduced != 1) {
        const uint64_t differenceMod = normalizeModulo(difference / static_cast<__int128>(gcd), rightReduced);
        const uint64_t inverseMod = normalizeModulo(extendedGcdCoefficient(leftReduced, rightReduced), rightReduced);
        leftStepCount = static_cast<uint64_t>(
            (static_cast<unsigned __int128>(differenceMod) * inverseMod) % rightReduced);
    }

    const unsigned __int128 period =
        static_cast<unsigned __int128>(left.stride) * static_cast<unsigned __int128>(rightReduced);
    unsigned __int128 solution =
        static_cast<unsigned __int128>(left.start) +
        static_cast<unsigned __int128>(left.stride) * leftStepCount;
    solution %= period;

    const unsigned __int128 lower128 = lower;
    if (solution < lower128) {
        solution += ((lower128 - solution + period - 1) / period) * period;
    }
    return solution <= static_cast<unsigned __int128>(upper);
}

json indexSetToJson(const ExampleIndexSet &indices) {
    if (!indices.isRangeBacked()) {
        return indices.materialize();
    }
    json ranges = json::array();
    for (const ExampleIndexRange &range : indices.getRanges()) {
        ranges.push_back(json{{"start", range.start}, {"count", range.count}, {"stride", range.stride}});
    }
    return json{{"ranges", std::move(ranges)}};
}

ExampleIndexSet readIndexSet(const json &partitions, const char *name) {
    if (!partitions.contains(name)) {
        throw std::runtime_error(std::string("DatasetSplitManifest is missing partition '") + name + "'.");
    }
    const json &value = partitions.at(name);
    if (value.is_array()) {
        return ExampleIndexSet(value.get<std::vector<uint64_t>>());
    }
    if (!value.is_object() || !value.contains("ranges") || !value.at("ranges").is_array()) {
        throw std::runtime_error(std::string("DatasetSplitManifest partition '") + name +
                                 "' must be an index array or an object containing ranges.");
    }
    std::vector<ExampleIndexRange> ranges;
    for (const json &range : value.at("ranges")) {
        if (!range.is_object()) {
            throw std::runtime_error(std::string("DatasetSplitManifest partition '") + name +
                                     "' range entries must be objects.");
        }
        ranges.push_back(ExampleIndexRange{.start = range.at("start").get<uint64_t>(),
                                           .count = range.at("count").get<uint64_t>(),
                                           .stride = range.value("stride", uint64_t{1})});
    }
    return ExampleIndexSet(std::move(ranges));
}

}  // namespace

uint64_t ExampleIndexRange::at(uint64_t position) const {
    if (position >= count) {
        throw std::out_of_range("ExampleIndexRange position is outside the range.");
    }
    return checkedAdd(start, checkedMul(position, stride, "ExampleIndexRange index"), "ExampleIndexRange index");
}

uint64_t ExampleIndexRange::last() const {
    if (count == 0) {
        throw std::runtime_error("ExampleIndexRange empty ranges do not have a last index.");
    }
    return at(count - 1);
}

ExampleIndexSet::ExampleIndexSet(std::vector<uint64_t> indices)
    : explicitIndices(std::make_shared<const std::vector<uint64_t>>(std::move(indices))),
      logicalSize(static_cast<uint64_t>(explicitIndices->size())) {}

ExampleIndexSet::ExampleIndexSet(std::vector<ExampleIndexRange> ranges) {
    auto storage = std::make_shared<RangeStorage>();
    storage->ranges = std::move(ranges);
    storage->cumulativeEnds.reserve(storage->ranges.size());

    for (size_t rangeOrdinal = 0; rangeOrdinal < storage->ranges.size(); ++rangeOrdinal) {
        const ExampleIndexRange &range = storage->ranges.at(rangeOrdinal);
        if (range.count == 0) {
            throw std::runtime_error("ExampleIndexSet ranges must have count >= 1.");
        }
        if (range.stride == 0) {
            throw std::runtime_error("ExampleIndexSet ranges must have stride >= 1.");
        }
        (void)range.last();
        for (size_t priorOrdinal = 0; priorOrdinal < rangeOrdinal; ++priorOrdinal) {
            if (rangesIntersect(storage->ranges.at(priorOrdinal), range)) {
                throw std::runtime_error("ExampleIndexSet ranges must not contain duplicate row indices.");
            }
        }
        logicalSize = checkedAdd(logicalSize, range.count, "ExampleIndexSet size");
        storage->cumulativeEnds.push_back(logicalSize);
    }
    rangeStorage = std::move(storage);
}

ExampleIndexSet ExampleIndexSet::contiguous(uint64_t start, uint64_t count) {
    if (count == 0) {
        return ExampleIndexSet(std::vector<ExampleIndexRange>{});
    }
    return ExampleIndexSet(std::vector<ExampleIndexRange>{{.start = start, .count = count, .stride = 1}});
}

ExampleIndexSet ExampleIndexSet::strided(uint64_t start, uint64_t count, uint64_t stride) {
    if (stride == 0) {
        throw std::runtime_error("ExampleIndexSet stride must be >= 1.");
    }
    if (count == 0) {
        return ExampleIndexSet(std::vector<ExampleIndexRange>{});
    }
    return ExampleIndexSet(std::vector<ExampleIndexRange>{{.start = start, .count = count, .stride = stride}});
}

uint64_t ExampleIndexSet::at(uint64_t position) const {
    if (position >= logicalSize) {
        throw std::out_of_range("ExampleIndexSet position is outside the set.");
    }
    if (explicitIndices != nullptr) {
        return explicitIndices->at(static_cast<size_t>(position));
    }
    const auto it = std::upper_bound(rangeStorage->cumulativeEnds.begin(), rangeStorage->cumulativeEnds.end(), position);
    const size_t rangeOrdinal = static_cast<size_t>(it - rangeStorage->cumulativeEnds.begin());
    const uint64_t rangeStartPosition = rangeOrdinal == 0 ? 0 : rangeStorage->cumulativeEnds.at(rangeOrdinal - 1);
    return rangeStorage->ranges.at(rangeOrdinal).at(position - rangeStartPosition);
}

const std::vector<ExampleIndexRange> &ExampleIndexSet::getRanges() const {
    if (rangeStorage == nullptr) {
        throw std::runtime_error("ExampleIndexSet is explicitly indexed rather than range-backed.");
    }
    return rangeStorage->ranges;
}

std::vector<uint64_t> ExampleIndexSet::materialize() const {
    if (explicitIndices != nullptr) {
        return *explicitIndices;
    }
    if (logicalSize > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::runtime_error("ExampleIndexSet is too large to materialize in this process.");
    }
    std::vector<uint64_t> out;
    out.reserve(static_cast<size_t>(logicalSize));
    for (const ExampleIndexRange &range : rangeStorage->ranges) {
        for (uint64_t position = 0; position < range.count; ++position) {
            out.push_back(range.at(position));
        }
    }
    return out;
}

bool ExampleIndexSet::operator==(const ExampleIndexSet &rhs) const {
    if (logicalSize != rhs.logicalSize) {
        return false;
    }
    if (explicitIndices != nullptr && rhs.explicitIndices != nullptr) {
        return *explicitIndices == *rhs.explicitIndices;
    }
    if (rangeStorage != nullptr && rhs.rangeStorage != nullptr && rangeStorage->ranges == rhs.rangeStorage->ranges) {
        return true;
    }
    for (uint64_t position = 0; position < logicalSize; ++position) {
        if (at(position) != rhs.at(position)) {
            return false;
        }
    }
    return true;
}

std::shared_ptr<const ExampleIndexSet> DatasetSplitManifest::makeIndexSet(ExampleIndexSet indices,
                                                                          uint64_t numExamples,
                                                                          const char *partitionName) {
    if (indices.isRangeBacked()) {
        for (const ExampleIndexRange &range : indices.getRanges()) {
            if (range.last() >= numExamples) {
                throw std::runtime_error(std::string("DatasetSplitManifest ") + partitionName +
                                         " contains row index outside dataset row count.");
            }
        }
    } else {
        std::unordered_set<uint64_t> unique;
        unique.reserve(static_cast<size_t>(indices.size()));
        for (uint64_t position = 0; position < indices.size(); ++position) {
            const uint64_t index = indices.at(position);
            if (index >= numExamples) {
                throw std::runtime_error(std::string("DatasetSplitManifest ") + partitionName +
                                         " contains row index outside dataset row count.");
            }
            if (!unique.insert(index).second) {
                throw std::runtime_error(std::string("DatasetSplitManifest ") + partitionName +
                                         " contains duplicate row index " + std::to_string(index) + ".");
            }
        }
    }
    return std::make_shared<const ExampleIndexSet>(std::move(indices));
}

DatasetSplitManifest::DatasetSplitManifest(const NamedDataset &dataset,
                                           std::vector<uint64_t> trainIndices,
                                           std::vector<uint64_t> validateIndices,
                                           std::optional<std::vector<uint64_t>> testIndices)
    : DatasetSplitManifest(dataset,
                           ExampleIndexSet(std::move(trainIndices)),
                           ExampleIndexSet(std::move(validateIndices)),
                           testIndices.has_value()
                               ? std::optional<ExampleIndexSet>(ExampleIndexSet(std::move(testIndices.value())))
                               : std::nullopt) {}

DatasetSplitManifest::DatasetSplitManifest(const NamedDataset &dataset,
                                           ExampleIndexSet trainIndices,
                                           ExampleIndexSet validateIndices,
                                           std::optional<ExampleIndexSet> testIndices)
    : DatasetSplitManifest(dataset.getId(),
                           dataset.getNumExamples(),
                           std::move(trainIndices),
                           std::move(validateIndices),
                           std::move(testIndices)) {}

DatasetSplitManifest::DatasetSplitManifest(DatasetId datasetId,
                                           uint64_t numExamples,
                                           ExampleIndexSet trainIndices,
                                           ExampleIndexSet validateIndices,
                                           std::optional<ExampleIndexSet> testIndices)
    : datasetId(std::move(datasetId)),
      numExamples(numExamples),
      train(makeIndexSet(std::move(trainIndices), numExamples, "train partition")),
      validate(makeIndexSet(std::move(validateIndices), numExamples, "validate partition")),
      explicitTestSplit(testIndices.has_value()) {
    if (explicitTestSplit) {
        test = makeIndexSet(std::move(testIndices.value()), numExamples, "test partition");
    } else {
        test = validate;
    }
}

void DatasetSplitManifest::validateAgainst(const NamedDataset &dataset) const {
    if (dataset.getId() != datasetId) {
        throw std::runtime_error("DatasetSplitManifest belongs to a different dataset identity.");
    }
    if (dataset.getNumExamples() != numExamples) {
        throw std::runtime_error("DatasetSplitManifest dataset row count does not match the opened dataset.");
    }
}

void DatasetSplitManifest::save(const std::filesystem::path &path) const {
    json manifest = {
        {"format", FORMAT},
        {"dataset_id", datasetId.str()},
        {"num_examples", numExamples},
        {"partitions",
         {{"train", indexSetToJson(*train)},
          {"validate", indexSetToJson(*validate)},
          {"test", explicitTestSplit ? indexSetToJson(*test) : json{{"alias", "validate"}}}}},
    };

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error("DatasetSplitManifest failed to open for writing: " + path.string());
    }
    out << manifest.dump(2) << '\n';
    if (!out.good()) {
        throw std::runtime_error("DatasetSplitManifest failed while writing: " + path.string());
    }
}

DatasetSplitManifest DatasetSplitManifest::load(const std::filesystem::path &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("DatasetSplitManifest failed to open: " + path.string());
    }

    json manifest;
    in >> manifest;
    if (!in.good() && !in.eof()) {
        throw std::runtime_error("DatasetSplitManifest failed while reading: " + path.string());
    }
    if (manifest.value("format", std::string()) != FORMAT) {
        throw std::runtime_error("DatasetSplitManifest has an unsupported format.");
    }

    DatasetId datasetId(manifest.at("dataset_id").get<std::string>());
    const uint64_t numExamples = manifest.at("num_examples").get<uint64_t>();
    const json &partitions = manifest.at("partitions");
    ExampleIndexSet trainIndices = readIndexSet(partitions, "train");
    ExampleIndexSet validateIndices = readIndexSet(partitions, "validate");

    std::optional<ExampleIndexSet> testIndices;
    if (!partitions.contains("test")) {
        throw std::runtime_error("DatasetSplitManifest is missing test partition metadata.");
    }
    const json &test = partitions.at("test");
    if (test.is_object() && test.contains("alias")) {
        if (test.value("alias", std::string()) != "validate") {
            throw std::runtime_error("DatasetSplitManifest test alias must reference validate.");
        }
    } else {
        json wrapped = json::object();
        wrapped["test"] = test;
        testIndices = readIndexSet(wrapped, "test");
    }

    return DatasetSplitManifest(std::move(datasetId),
                                numExamples,
                                std::move(trainIndices),
                                std::move(validateIndices),
                                std::move(testIndices));
}

bool DatasetSplitManifest::operator==(const DatasetSplitManifest &rhs) const {
    return datasetId == rhs.datasetId && numExamples == rhs.numExamples && *train == *rhs.train &&
           *validate == *rhs.validate && *test == *rhs.test && explicitTestSplit == rhs.explicitTestSplit;
}

}  // namespace Thor
