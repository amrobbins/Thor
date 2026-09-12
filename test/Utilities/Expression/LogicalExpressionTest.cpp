#include "Utilities/Expression/LogicalExpression.h"
#include "Utilities/Expression/CudaKernelExpression.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace ThorImplementation;

namespace {

template <typename T>
concept HasPublicPhysicalNodeImport = requires(T& importer) {
    importer.importNode(uint32_t{0});
};

static_assert(!HasPublicPhysicalNodeImport<LogicalExpressionImporter>,
              "Physical->logical import must remain graph-scoped through importOutputs().");

size_t countUniqueLogicalNodes(const std::vector<LogicalExpression>& roots) {
    std::unordered_set<const LogicalExpressionNode*> seen;
    std::function<void(const LogicalExpression&)> visit = [&](const LogicalExpression& node) {
        if (!node || !seen.insert(node.get()).second) {
            return;
        }
        for (const LogicalDependency& dependency : node->dependencies()) {
            visit(dependency.node);
        }
        if (node->cudaKernelApplication()) {
            for (const LogicalExpression& input : node->cudaKernelApplication()->inputs()) {
                visit(input);
            }
        }
    };
    for (const LogicalExpression& root : roots) {
        visit(root);
    }
    return seen.size();
}

TEST(LogicalExpressionCore, ReusingOneHandleCreatesOneSharedDependency) {
    const LogicalExpression x = makeLogicalInput("x");
    const LogicalExpression w = makeLogicalInput("w");
    const LogicalExpression z = makeLogicalBinary(ExprOp::MATMUL, x, w);
    const LogicalExpression a = makeLogicalUnary(ExprOp::SIN, z);
    const LogicalExpression b = makeLogicalUnary(ExprOp::COS, z);

    EXPECT_EQ(a->dependency(LogicalDependencyKind::Lhs).get(), z.get());
    EXPECT_EQ(b->dependency(LogicalDependencyKind::Lhs).get(), z.get());
    EXPECT_EQ(a->dependency(LogicalDependencyKind::Lhs).get(), b->dependency(LogicalDependencyKind::Lhs).get());
    EXPECT_EQ(countUniqueLogicalNodes({a, b}), 5u);
}

TEST(LogicalExpressionCore, SeparatelyAuthoredEquivalentOperationsRemainDistinct) {
    const LogicalExpression x = makeLogicalInput("x");
    const LogicalExpression w = makeLogicalInput("w");
    const LogicalExpression z1 = makeLogicalBinary(ExprOp::MATMUL, x, w);
    const LogicalExpression z2 = makeLogicalBinary(ExprOp::MATMUL, x, w);

    EXPECT_NE(z1.get(), z2.get());
    EXPECT_EQ(z1->dependency(LogicalDependencyKind::Lhs).get(), x.get());
    EXPECT_EQ(z2->dependency(LogicalDependencyKind::Lhs).get(), x.get());
    EXPECT_EQ(countUniqueLogicalNodes({z1, z2}), 4u);
}

TEST(LogicalExpressionCore, DerivedNodesCannotMutateTheirParentsOrSiblingBranches) {
    static_assert(std::is_same_v<LogicalExpression, std::shared_ptr<const LogicalExpressionNode>>);
    static_assert(!std::is_copy_assignable_v<LogicalExpressionNode>);
    static_assert(!std::is_move_assignable_v<LogicalExpressionNode>);

    const LogicalExpression x = makeLogicalInput("x");
    const LogicalExpression sin_x = makeLogicalUnary(ExprOp::SIN, x);
    const LogicalExpression cos_x = makeLogicalUnary(ExprOp::COS, x);
    const LogicalExpression tan_sin_x = makeLogicalUnary(ExprOp::TAN, sin_x);

    EXPECT_EQ(x->op(), ExprOp::INPUT);
    EXPECT_TRUE(x->dependencies().empty());
    EXPECT_EQ(sin_x->op(), ExprOp::SIN);
    EXPECT_EQ(sin_x->dependency(LogicalDependencyKind::Lhs).get(), x.get());
    EXPECT_EQ(cos_x->op(), ExprOp::COS);
    EXPECT_EQ(cos_x->dependency(LogicalDependencyKind::Lhs).get(), x.get());
    EXPECT_EQ(tan_sin_x->dependency(LogicalDependencyKind::Lhs).get(), sin_x.get());
}

TEST(LogicalExpressionCore, AncestrySurvivesIntermediateHandleDestruction) {
    const LogicalExpression x = makeLogicalInput("x");
    const LogicalExpression w = makeLogicalInput("w");
    LogicalExpression branch;
    const LogicalExpressionNode* producer_identity = nullptr;
    {
        LogicalExpression producer = makeLogicalBinary(ExprOp::MATMUL, x, w);
        producer_identity = producer.get();
        branch = makeLogicalUnary(ExprOp::SIN, producer);
    }

    ASSERT_NE(branch, nullptr);
    EXPECT_EQ(branch->dependency(LogicalDependencyKind::Lhs).get(), producer_identity);
    EXPECT_EQ(branch->dependency(LogicalDependencyKind::Lhs)->op(), ExprOp::MATMUL);
}

TEST(LogicalExpressionCore, MultipleRootsSafelyShareArbitraryAncestry) {
    const LogicalExpression x = makeLogicalInput("x");
    const LogicalExpression w = makeLogicalInput("w");
    const LogicalExpression z = makeLogicalBinary(ExprOp::MATMUL, x, w);
    const LogicalExpression shared = makeLogicalUnary(ExprOp::TANH, z);
    const LogicalExpression left = makeLogicalBinary(ExprOp::ADD, shared, makeLogicalScalar(1.0));
    const LogicalExpression right = makeLogicalBinary(ExprOp::MUL, shared, makeLogicalScalar(2.0));

    EXPECT_EQ(left->dependency(LogicalDependencyKind::Lhs).get(), shared.get());
    EXPECT_EQ(right->dependency(LogicalDependencyKind::Lhs).get(), shared.get());
    EXPECT_EQ(countUniqueLogicalNodes({left, right}), 8u);
}

TEST(LogicalExpressionCore, RejectsPhysicalIndicesInSemanticSnapshot) {
    ExprNode semantics{};
    semantics.op = ExprOp::SIN;
    semantics.lhs = 3;
    const LogicalExpression x = makeLogicalInput("x");

    EXPECT_THROW((void)LogicalExpressionNode::create(
                     semantics, {{LogicalDependencyKind::Lhs, 0, x}}),
                 std::invalid_argument);
}


size_t countPhysicalOp(const PhysicalOutputs& outputs, ExprOp op) {
    return static_cast<size_t>(std::count_if(outputs.expr->nodes.begin(), outputs.expr->nodes.end(),
                                             [op](const ExprNode& node) { return node.op == op; }));
}

TEST(LogicalExpressionLowering, SharedAncestryLowersExactlyOnceAcrossMultipleRoots) {
    const LogicalExpression x = makeLogicalInput("x");
    const LogicalExpression w = makeLogicalInput("w");
    const LogicalExpression z = makeLogicalBinary(ExprOp::MATMUL, x, w);
    const LogicalExpression a = makeLogicalUnary(ExprOp::SIN, z);
    const LogicalExpression b = makeLogicalUnary(ExprOp::COS, z);

    const PhysicalOutputs lowered = LogicalExpressionLowerer::lower({{"a", a, {}}, {"b", b, {}}});
    ASSERT_TRUE(lowered.expr);
    EXPECT_EQ(lowered.expr->nodes.size(), 5u);
    EXPECT_EQ(countPhysicalOp(lowered, ExprOp::MATMUL), 1u);
    ASSERT_EQ(lowered.outputs.size(), 2u);
    const ExprNode& sin_node = lowered.expr->nodes.at(lowered.outputs[0].node_idx);
    const ExprNode& cos_node = lowered.expr->nodes.at(lowered.outputs[1].node_idx);
    EXPECT_EQ(sin_node.lhs, cos_node.lhs);
    EXPECT_EQ(lowered.expr->nodes.at(sin_node.lhs).op, ExprOp::MATMUL);
}

TEST(LogicalExpressionLowering, IndependentlyAuthoredEquivalentProducersStayDistinct) {
    const LogicalExpression x = makeLogicalInput("x");
    const LogicalExpression w = makeLogicalInput("w");
    const LogicalExpression z1 = makeLogicalBinary(ExprOp::MATMUL, x, w);
    const LogicalExpression z2 = makeLogicalBinary(ExprOp::MATMUL, x, w);

    const PhysicalOutputs lowered = LogicalExpressionLowerer::lower({{"z1", z1, {}}, {"z2", z2, {}}});
    EXPECT_EQ(countPhysicalOp(lowered, ExprOp::MATMUL), 2u);
    ASSERT_EQ(lowered.outputs.size(), 2u);
    EXPECT_NE(lowered.outputs[0].node_idx, lowered.outputs[1].node_idx);
}

TEST(LogicalExpressionLowering, DeepChainAddsOnePhysicalNodePerLogicalOperation) {
    constexpr size_t steps = 64;
    LogicalExpression current = makeLogicalInput("x");
    for (size_t i = 0; i < steps; ++i) {
        current = makeLogicalUnary(i % 2 == 0 ? ExprOp::SIN : ExprOp::COS, current);
    }

    const PhysicalOutputs lowered = LogicalExpressionLowerer::lower({{"y", current, {}}});
    EXPECT_EQ(lowered.expr->nodes.size(), steps + 1);
}

TEST(LogicalExpressionLowering, FanOutAndRecombinePreservesOneSharedProducer) {
    const LogicalExpression x = makeLogicalInput("x");
    const LogicalExpression w = makeLogicalInput("w");
    const LogicalExpression z = makeLogicalBinary(ExprOp::MATMUL, x, w);
    const LogicalExpression left = makeLogicalUnary(ExprOp::SIN, z);
    const LogicalExpression right = makeLogicalUnary(ExprOp::COS, z);
    const LogicalExpression y = makeLogicalBinary(ExprOp::ADD, left, right);

    const PhysicalOutputs lowered = LogicalExpressionLowerer::lower({{"y", y, {}}});
    EXPECT_EQ(lowered.expr->nodes.size(), 6u);
    EXPECT_EQ(countPhysicalOp(lowered, ExprOp::MATMUL), 1u);
    const ExprNode& add = lowered.expr->nodes.at(lowered.outputs.front().node_idx);
    EXPECT_EQ(lowered.expr->nodes.at(add.lhs).lhs, lowered.expr->nodes.at(add.rhs).lhs);
}

TEST(LogicalExpressionLowering, RepeatedLoweringIsDeterministicIncludingAttributesAndOutputContracts) {
    const LogicalExpression x = makeLogicalInput("x", NamedInput::Kind::Tensor, DataType::FP16, DataType::FP32, DataType::FP16);
    const LogicalExpression w = makeLogicalInput("w", NamedInput::Kind::Tensor, DataType::FP16, DataType::FP32, DataType::FP16);
    ExprNode matmul_semantics{};
    matmul_semantics.op = ExprOp::MATMUL;
    matmul_semantics.transpose_rhs = true;
    matmul_semantics.compute_dtype = DataType::FP32;
    matmul_semantics.output_dtype = DataType::FP16;
    const LogicalExpression z = makeLogicalBinary(ExprOp::MATMUL, x, w, matmul_semantics);
    const LogicalExpression y = makeLogicalUnary(ExprOp::TANH, z);
    OutputMaterializationContract contract;
    contract.storage_dtype = DataType::FP16;
    contract.require_distinct_storage = true;

    const std::vector<LogicalNamedOutput> roots{{"z", z, {}}, {"y", y, contract}};
    const PhysicalOutputs first = LogicalExpressionLowerer::lower(roots);
    const PhysicalOutputs second = LogicalExpressionLowerer::lower(roots);

    ASSERT_TRUE(first.expr);
    ASSERT_TRUE(second.expr);
    ASSERT_EQ(first.expr->inputs.size(), second.expr->inputs.size());
    for (size_t i = 0; i < first.expr->inputs.size(); ++i) {
        EXPECT_EQ(first.expr->inputs[i].name, second.expr->inputs[i].name);
        EXPECT_EQ(first.expr->inputs[i].slot, second.expr->inputs[i].slot);
        EXPECT_EQ(first.expr->inputs[i].kind, second.expr->inputs[i].kind);
    }
    ASSERT_EQ(first.outputs.size(), second.outputs.size());
    for (size_t i = 0; i < first.outputs.size(); ++i) {
        EXPECT_EQ(first.outputs[i].name, second.outputs[i].name);
        EXPECT_EQ(first.outputs[i].node_idx, second.outputs[i].node_idx);
        EXPECT_EQ(first.outputs[i].materialization, second.outputs[i].materialization);
    }
    EXPECT_EQ(canonicalize(first), canonicalize(second));
    ASSERT_EQ(first.expr->nodes.size(), second.expr->nodes.size());
    for (size_t i = 0; i < first.expr->nodes.size(); ++i) {
        EXPECT_EQ(first.expr->nodes[i].op, second.expr->nodes[i].op);
        EXPECT_EQ(first.expr->nodes[i].lhs, second.expr->nodes[i].lhs);
        EXPECT_EQ(first.expr->nodes[i].rhs, second.expr->nodes[i].rhs);
        EXPECT_EQ(first.expr->nodes[i].aux, second.expr->nodes[i].aux);
        EXPECT_EQ(first.expr->nodes[i].transpose_rhs, second.expr->nodes[i].transpose_rhs);
        EXPECT_EQ(first.expr->nodes[i].compute_dtype, second.expr->nodes[i].compute_dtype);
        EXPECT_EQ(first.expr->nodes[i].output_dtype, second.expr->nodes[i].output_dtype);
    }
}


PhysicalOutputs makeSharedPhysicalImportFixture(bool duplicate_matmul = false) {
    auto expr = std::make_shared<PhysicalExpression>();
    const uint32_t x_slot = expr->getOrCreateInputSlot("x", NamedInput::Kind::Tensor);
    const uint32_t w_slot = expr->getOrCreateInputSlot("w", NamedInput::Kind::Tensor);

    ExprNode x{};
    x.op = ExprOp::INPUT;
    x.input_slot = x_slot;
    x.input_tensor_dtype = DataType::FP16;
    x.compute_dtype = DataType::FP32;
    x.output_dtype = DataType::FP16;
    expr->nodes.push_back(x);

    ExprNode w = x;
    w.input_slot = w_slot;
    expr->nodes.push_back(w);

    ExprNode matmul{};
    matmul.op = ExprOp::MATMUL;
    matmul.lhs = 0;
    matmul.rhs = 1;
    matmul.transpose_rhs = true;
    matmul.compute_dtype = DataType::FP32;
    matmul.output_dtype = DataType::FP16;
    expr->nodes.push_back(matmul);
    const uint32_t first_matmul = 2;

    uint32_t second_matmul = first_matmul;
    if (duplicate_matmul) {
        expr->nodes.push_back(matmul);
        second_matmul = 3;
    }

    ExprNode sin{};
    sin.op = ExprOp::SIN;
    sin.lhs = first_matmul;
    expr->nodes.push_back(sin);
    const uint32_t sin_index = static_cast<uint32_t>(expr->nodes.size() - 1);

    ExprNode cos{};
    cos.op = ExprOp::COS;
    cos.lhs = second_matmul;
    expr->nodes.push_back(cos);
    const uint32_t cos_index = static_cast<uint32_t>(expr->nodes.size() - 1);

    OutputMaterializationContract materialization;
    materialization.storage_dtype = DataType::FP16;
    return PhysicalOutputs{
        .expr = std::move(expr),
        .outputs = {{"sin", sin_index, materialization}, {"cos", cos_index, {}}},
    };
}

TEST(LogicalExpressionImport, SharedPhysicalAncestorBecomesOneLogicalNodeAcrossRoots) {
    const PhysicalOutputs physical = makeSharedPhysicalImportFixture(false);
    LogicalExpressionImporter importer(*physical.expr);
    const std::vector<LogicalNamedOutput> roots = importer.importOutputs(physical.outputs);

    ASSERT_EQ(roots.size(), 2u);
    const LogicalExpression& sin_parent = roots[0].node->dependency(LogicalDependencyKind::Lhs);
    const LogicalExpression& cos_parent = roots[1].node->dependency(LogicalDependencyKind::Lhs);
    EXPECT_EQ(sin_parent.get(), cos_parent.get());
    EXPECT_EQ(sin_parent->op(), ExprOp::MATMUL);
    EXPECT_EQ(roots[0].materialization, physical.outputs[0].materialization);
}

TEST(LogicalExpressionImport, StructurallyEquivalentDistinctPhysicalNodesRemainDistinct) {
    const PhysicalOutputs physical = makeSharedPhysicalImportFixture(true);
    LogicalExpressionImporter importer(*physical.expr);
    const std::vector<LogicalNamedOutput> roots = importer.importOutputs(physical.outputs);

    const LogicalExpression& sin_parent = roots[0].node->dependency(LogicalDependencyKind::Lhs);
    const LogicalExpression& cos_parent = roots[1].node->dependency(LogicalDependencyKind::Lhs);
    EXPECT_NE(sin_parent.get(), cos_parent.get());
    EXPECT_EQ(sin_parent->op(), ExprOp::MATMUL);
    EXPECT_EQ(cos_parent->op(), ExprOp::MATMUL);
}

TEST(LogicalExpressionImport, InputBindingsAndSemanticAttributesSurviveImport) {
    const PhysicalOutputs physical = makeSharedPhysicalImportFixture(false);
    LogicalExpressionImporter importer(*physical.expr);
    const std::vector<LogicalNamedOutput> roots = importer.importOutputs(physical.outputs);
    const LogicalExpression matmul = roots[0].node->dependency(LogicalDependencyKind::Lhs);
    const LogicalExpression lhs = matmul->dependency(LogicalDependencyKind::Lhs);
    const LogicalExpression rhs = matmul->dependency(LogicalDependencyKind::Rhs);

    ASSERT_TRUE(lhs->inputBinding());
    ASSERT_TRUE(rhs->inputBinding());
    EXPECT_EQ(lhs->inputBinding()->name, "x");
    EXPECT_EQ(rhs->inputBinding()->name, "w");
    EXPECT_EQ(lhs->inputBinding()->kind, NamedInput::Kind::Tensor);
    EXPECT_EQ(lhs->semantics().input_tensor_dtype, DataType::FP16);
    EXPECT_EQ(lhs->semantics().compute_dtype, DataType::FP32);
    EXPECT_TRUE(matmul->semantics().transpose_rhs);
    EXPECT_EQ(matmul->semantics().compute_dtype, DataType::FP32);
    EXPECT_EQ(matmul->semantics().output_dtype, DataType::FP16);
}

TEST(LogicalExpressionImport, ImportDoesNotMutatePhysicalSource) {
    const PhysicalOutputs physical = makeSharedPhysicalImportFixture(false);
    const std::string canonical_before = canonicalize(physical);
    const size_t node_count_before = physical.expr->nodes.size();
    const size_t input_count_before = physical.expr->inputs.size();

    LogicalExpressionImporter importer(*physical.expr);
    (void)importer.importOutputs(physical.outputs);

    EXPECT_EQ(physical.expr->nodes.size(), node_count_before);
    EXPECT_EQ(physical.expr->inputs.size(), input_count_before);
    EXPECT_EQ(canonicalize(physical), canonical_before);
}

TEST(LogicalExpressionImport, ImportThenLowerPreservesSharedSourceTopology) {
    const PhysicalOutputs physical = makeSharedPhysicalImportFixture(false);
    LogicalExpressionImporter importer(*physical.expr);
    const std::vector<LogicalNamedOutput> roots = importer.importOutputs(physical.outputs);
    const PhysicalOutputs round_trip = LogicalExpressionLowerer::lower(roots);

    EXPECT_EQ(round_trip.expr->nodes.size(), physical.expr->nodes.size());
    EXPECT_EQ(countPhysicalOp(round_trip, ExprOp::MATMUL), 1u);
    const ExprNode& sin_node = round_trip.expr->nodes.at(round_trip.outputs[0].node_idx);
    const ExprNode& cos_node = round_trip.expr->nodes.at(round_trip.outputs[1].node_idx);
    EXPECT_EQ(sin_node.lhs, cos_node.lhs);
    EXPECT_EQ(round_trip.outputs[0].materialization, physical.outputs[0].materialization);
}


uint32_t appendPhysicalInput(PhysicalExpression& expr,
                             const std::string& name,
                             NamedInput::Kind kind = NamedInput::Kind::Tensor,
                             DataType dtype = DataType::FP32) {
    ExprNode node{};
    switch (kind) {
        case NamedInput::Kind::Tensor: node.op = ExprOp::INPUT; break;
        case NamedInput::Kind::RuntimeScalarFp32: node.op = ExprOp::RUNTIME_SCALAR; break;
        case NamedInput::Kind::TensorRuntimeScalar: node.op = ExprOp::TENSOR_RUNTIME_SCALAR; break;
    }
    node.input_slot = expr.getOrCreateInputSlot(name, kind);
    node.input_tensor_dtype = dtype;
    node.compute_dtype = dtype;
    node.output_dtype = dtype;
    const uint32_t index = static_cast<uint32_t>(expr.nodes.size());
    expr.nodes.push_back(std::move(node));
    return index;
}

PhysicalOutputs singleRootPhysicalForOp(ExprOp op) {
    auto expr = std::make_shared<PhysicalExpression>();
    if (op == ExprOp::INPUT || op == ExprOp::RUNTIME_SCALAR || op == ExprOp::TENSOR_RUNTIME_SCALAR) {
        NamedInput::Kind kind = NamedInput::Kind::Tensor;
        if (op == ExprOp::RUNTIME_SCALAR) kind = NamedInput::Kind::RuntimeScalarFp32;
        if (op == ExprOp::TENSOR_RUNTIME_SCALAR) kind = NamedInput::Kind::TensorRuntimeScalar;
        const uint32_t root = appendPhysicalInput(*expr, "root", kind);
        return PhysicalOutputs{.expr = std::move(expr), .outputs = {{"y", root, {}}}};
    }

    ExprNode node{};
    node.op = op;
    if (Expression::isUnaryOp(op)) {
        node.lhs = appendPhysicalInput(*expr, "a");
    } else if (Expression::isBinaryOp(op)) {
        node.lhs = appendPhysicalInput(*expr, "a");
        node.rhs = appendPhysicalInput(*expr, "b");
    } else if (Expression::isTernaryOp(op)) {
        node.lhs = appendPhysicalInput(*expr, "a");
        node.rhs = appendPhysicalInput(*expr, "b");
        node.aux = appendPhysicalInput(*expr, "c");
    } else if (Expression::isLeafOp(op)) {
        if (op == ExprOp::SCALAR_FP) node.scalar_fp = 3.25;
        if (op == ExprOp::FILL) node.fill_dims = {2, 3};
    } else {
        throw std::runtime_error("singleRootPhysicalForOp received unsupported special operation");
    }
    const uint32_t root = static_cast<uint32_t>(expr->nodes.size());
    expr->nodes.push_back(std::move(node));
    return PhysicalOutputs{.expr = std::move(expr), .outputs = {{"y", root, {}}}};
}

TEST(LogicalExpressionRoundTrip, EveryCurrentExprOpHasARepresentableLogicalArity) {
    const uint16_t first = static_cast<uint16_t>(ExprOp::INPUT);
    const uint16_t last = static_cast<uint16_t>(ExprOp::RAGGED_SOFTMAX_BACKWARD);
    size_t covered = 0;
    for (uint16_t raw = first; raw <= last; ++raw) {
        const ExprOp op = static_cast<ExprOp>(raw);
        if (op == ExprOp::CUDA_KERNEL_OUTPUT) {
            continue;
        }
        SCOPED_TRACE(static_cast<int>(raw));
        const PhysicalOutputs source = singleRootPhysicalForOp(op);
        LogicalExpressionImporter importer(*source.expr);
        const std::vector<LogicalNamedOutput> logical = importer.importOutputs(source.outputs);
        ASSERT_EQ(logical.size(), 1u);
        EXPECT_EQ(logical.front().node->op(), op);
        const PhysicalOutputs round_trip = LogicalExpressionLowerer::lower(logical);
        ASSERT_EQ(round_trip.outputs.size(), 1u);
        EXPECT_EQ(round_trip.expr->nodes.at(round_trip.outputs.front().node_idx).op, op);
        ++covered;
    }
    EXPECT_EQ(covered, static_cast<size_t>(last - first));
}

TEST(LogicalExpressionRoundTrip, GemmOptionalExpressionOperandsPreserveIdentityAndRoles) {
    auto expr = std::make_shared<PhysicalExpression>();
    const uint32_t a = appendPhysicalInput(*expr, "a");
    const uint32_t b = appendPhysicalInput(*expr, "b");
    const uint32_t c = appendPhysicalInput(*expr, "c");
    const uint32_t scale = appendPhysicalInput(*expr, "scale");
    const uint32_t epilogue = appendPhysicalInput(*expr, "epilogue");

    ExprNode gemm{};
    gemm.op = ExprOp::GEMM;
    gemm.lhs = a;
    gemm.rhs = b;
    gemm.aux = c;
    gemm.alpha_node = scale;
    gemm.beta_node = scale;
    gemm.matmul_backward_epilogue = MatmulBackwardEpilogue::DGelu;
    gemm.matmul_epilogue_aux = epilogue;
    gemm.transpose_lhs = true;
    gemm.transpose_aux = true;
    const uint32_t root = static_cast<uint32_t>(expr->nodes.size());
    expr->nodes.push_back(gemm);
    const PhysicalOutputs source{.expr = expr, .outputs = {{"y", root, {}}}};

    LogicalExpressionImporter importer(*source.expr);
    const LogicalExpression logical = importer.importOutputs(source.outputs).front().node;
    EXPECT_EQ(logical->dependency(LogicalDependencyKind::Alpha).get(), logical->dependency(LogicalDependencyKind::Beta).get());
    EXPECT_EQ(logical->semantics().matmul_backward_epilogue, MatmulBackwardEpilogue::DGelu);
    EXPECT_TRUE(logical->semantics().transpose_lhs);
    EXPECT_TRUE(logical->semantics().transpose_aux);
    EXPECT_NO_THROW(validateLogicalGraph({{"y", logical, {}}}));

    const PhysicalOutputs round_trip = LogicalExpressionLowerer::lower({{"y", logical, {}}});
    const ExprNode& result = round_trip.expr->nodes.at(round_trip.outputs.front().node_idx);
    EXPECT_EQ(result.alpha_node, result.beta_node);
    EXPECT_NE(result.matmul_epilogue_aux, UINT32_MAX);
    EXPECT_EQ(result.matmul_backward_epilogue, MatmulBackwardEpilogue::DGelu);
}

TEST(LogicalExpressionRoundTrip, AttentionOptionalMetadataDependenciesRemainSharedAndOrdered) {
    auto expr = std::make_shared<PhysicalExpression>();
    const uint32_t q = appendPhysicalInput(*expr, "q");
    const uint32_t k = appendPhysicalInput(*expr, "k");
    const uint32_t v = appendPhysicalInput(*expr, "v");
    const uint32_t bias = appendPhysicalInput(*expr, "bias");
    const uint32_t shared_length = appendPhysicalInput(*expr, "length");
    const uint32_t offsets_q = appendPhysicalInput(*expr, "offsets_q");
    const uint32_t offsets_kv = appendPhysicalInput(*expr, "offsets_kv");
    const uint32_t page_k = appendPhysicalInput(*expr, "page_k");
    const uint32_t page_v = appendPhysicalInput(*expr, "page_v");
    const uint32_t seed = appendPhysicalInput(*expr, "seed", NamedInput::Kind::TensorRuntimeScalar, DataType::UINT64);
    const uint32_t dropout_offset = appendPhysicalInput(*expr, "dropout_offset", NamedInput::Kind::TensorRuntimeScalar, DataType::UINT64);
    const uint32_t descale_q = appendPhysicalInput(*expr, "descale_q");
    const uint32_t descale_k = appendPhysicalInput(*expr, "descale_k");
    const uint32_t descale_v = appendPhysicalInput(*expr, "descale_v");
    const uint32_t descale_s = appendPhysicalInput(*expr, "descale_s");
    const uint32_t scale_s = appendPhysicalInput(*expr, "scale_s");
    const uint32_t scale_o = appendPhysicalInput(*expr, "scale_o");
    const uint32_t amax_s = appendPhysicalInput(*expr, "amax_s");
    const uint32_t amax_o = appendPhysicalInput(*expr, "amax_o");

    ExprNode attention{};
    attention.op = ExprOp::ATTENTION;
    attention.lhs = q;
    attention.rhs = k;
    attention.aux = v;
    attention.alpha_node = bias;
    attention.attention_use_bias = true;
    attention.attention_use_padding_mask = true;
    attention.attention_seq_len_q_node = shared_length;
    attention.attention_seq_len_kv_node = shared_length;
    attention.attention_use_ragged_offsets = true;
    attention.attention_ragged_offset_q_node = offsets_q;
    attention.attention_ragged_offset_kv_node = offsets_kv;
    attention.attention_use_paged_kv_cache = true;
    attention.attention_page_table_k_node = page_k;
    attention.attention_page_table_v_node = page_v;
    attention.attention_dropout_probability = 0.125f;
    attention.attention_dropout_seed_node = seed;
    attention.attention_dropout_offset_node = dropout_offset;
    attention.attention_use_fp8_forward_scaling = true;
    attention.attention_descale_q_node = descale_q;
    attention.attention_descale_k_node = descale_k;
    attention.attention_descale_v_node = descale_v;
    attention.attention_descale_s_node = descale_s;
    attention.attention_scale_s_node = scale_s;
    attention.attention_scale_o_node = scale_o;
    attention.attention_amax_s_node = amax_s;
    attention.attention_amax_o_node = amax_o;
    const uint32_t root = static_cast<uint32_t>(expr->nodes.size());
    expr->nodes.push_back(attention);

    LogicalExpressionImporter importer(*expr);
    const LogicalExpression logical = importer.importOutputs({{"y", root, {}}}).front().node;
    EXPECT_EQ(logical->dependency(LogicalDependencyKind::AttentionSeqLenQ).get(),
              logical->dependency(LogicalDependencyKind::AttentionSeqLenKv).get());
    EXPECT_TRUE(logical->semantics().attention_use_padding_mask);
    EXPECT_TRUE(logical->semantics().attention_use_ragged_offsets);
    EXPECT_TRUE(logical->semantics().attention_use_paged_kv_cache);
    EXPECT_TRUE(logical->semantics().attention_use_fp8_forward_scaling);
    EXPECT_NO_THROW(validateLogicalGraph({{"y", logical, {}}}));

    const PhysicalOutputs round_trip = LogicalExpressionLowerer::lower({{"y", logical, {}}});
    const ExprNode& result = round_trip.expr->nodes.at(round_trip.outputs.front().node_idx);
    EXPECT_EQ(result.attention_seq_len_q_node, result.attention_seq_len_kv_node);
    EXPECT_NE(result.attention_ragged_offset_q_node, UINT32_MAX);
    EXPECT_NE(result.attention_ragged_offset_kv_node, UINT32_MAX);
    EXPECT_NE(result.attention_page_table_k_node, UINT32_MAX);
    EXPECT_NE(result.attention_dropout_seed_node, UINT32_MAX);
    EXPECT_NE(result.attention_amax_o_node, UINT32_MAX);
}

TEST(LogicalExpressionRoundTrip, AttentionBackwardUpstreamGradientAndBiasDependenciesSurvive) {
    auto expr = std::make_shared<PhysicalExpression>();
    const uint32_t q = appendPhysicalInput(*expr, "q");
    const uint32_t k = appendPhysicalInput(*expr, "k");
    const uint32_t v = appendPhysicalInput(*expr, "v");
    const uint32_t dO = appendPhysicalInput(*expr, "dO");
    const uint32_t bias = appendPhysicalInput(*expr, "bias");
    const uint32_t shared_length = appendPhysicalInput(*expr, "length", NamedInput::Kind::Tensor, DataType::INT32);

    ExprNode backward{};
    backward.op = ExprOp::ATTENTION_BACKWARD_Q;
    backward.lhs = q;
    backward.rhs = k;
    backward.aux = v;
    backward.alpha_node = dO;
    backward.beta_node = bias;
    backward.attention_use_bias = true;
    backward.attention_use_padding_mask = true;
    backward.attention_seq_len_q_node = shared_length;
    backward.attention_seq_len_kv_node = shared_length;
    const uint32_t root = static_cast<uint32_t>(expr->nodes.size());
    expr->nodes.push_back(backward);

    LogicalExpressionImporter importer(*expr);
    const LogicalExpression logical = importer.importOutputs({{"y", root, {}}}).front().node;
    EXPECT_EQ(logical->dependency(LogicalDependencyKind::Alpha)->inputBinding()->name, "dO");
    EXPECT_EQ(logical->dependency(LogicalDependencyKind::Beta)->inputBinding()->name, "bias");
    EXPECT_EQ(logical->dependency(LogicalDependencyKind::AttentionSeqLenQ).get(),
              logical->dependency(LogicalDependencyKind::AttentionSeqLenKv).get());
    EXPECT_NO_THROW(validateLogicalGraph({{"dq", logical, {}}}));

    const PhysicalOutputs round_trip = LogicalExpressionLowerer::lower({{"dq", logical, {}}});
    const ExprNode& result = round_trip.expr->nodes.at(round_trip.outputs.front().node_idx);
    EXPECT_NE(result.alpha_node, UINT32_MAX);
    EXPECT_NE(result.beta_node, UINT32_MAX);
    EXPECT_EQ(result.attention_seq_len_q_node, result.attention_seq_len_kv_node);
    EXPECT_TRUE(result.attention_use_bias);
    EXPECT_TRUE(result.attention_use_padding_mask);
}

TEST(LogicalExpressionRoundTrip, RopeMetadataAndSemanticVectorsSurvive) {
    auto expr = std::make_shared<PhysicalExpression>();
    const uint32_t value = appendPhysicalInput(*expr, "value");
    const uint32_t length = appendPhysicalInput(*expr, "length");
    const uint32_t positions = appendPhysicalInput(*expr, "positions");
    ExprNode rope{};
    rope.op = ExprOp::ROPE;
    rope.lhs = value;
    rope.rope_effective_sequence_length_node = length;
    rope.rope_position_ids_node = positions;
    rope.rope_sequence_axis = 1;
    rope.rope_head_dim_axis = 2;
    rope.rope_rotary_dim = 64;
    rope.rope_scaling_kind = RotaryScalingKind::LongRope;
    rope.rope_scaling_factor = 2.5;
    rope.rope_original_max_position_embeddings = 8192;
    rope.rope_long_rope_short_factors = {1.0, 1.25};
    rope.rope_long_rope_long_factors = {2.0, 3.0};
    const uint32_t root = static_cast<uint32_t>(expr->nodes.size());
    expr->nodes.push_back(rope);

    LogicalExpressionImporter importer(*expr);
    const LogicalExpression logical = importer.importOutputs({{"y", root, {}}}).front().node;
    EXPECT_EQ(logical->dependency(LogicalDependencyKind::RopeEffectiveSequenceLength)->inputBinding()->name, "length");
    EXPECT_EQ(logical->dependency(LogicalDependencyKind::RopePositionIds)->inputBinding()->name, "positions");
    EXPECT_EQ(logical->semantics().rope_long_rope_short_factors, rope.rope_long_rope_short_factors);
    EXPECT_EQ(logical->semantics().rope_long_rope_long_factors, rope.rope_long_rope_long_factors);
    EXPECT_NO_THROW(validateLogicalGraph({{"y", logical, {}}}));

    const PhysicalOutputs round_trip = LogicalExpressionLowerer::lower({{"y", logical, {}}});
    const ExprNode& result = round_trip.expr->nodes.at(round_trip.outputs.front().node_idx);
    EXPECT_NE(result.rope_effective_sequence_length_node, UINT32_MAX);
    EXPECT_NE(result.rope_position_ids_node, UINT32_MAX);
    EXPECT_EQ(result.rope_scaling_kind, RotaryScalingKind::LongRope);
    EXPECT_EQ(result.rope_long_rope_long_factors, rope.rope_long_rope_long_factors);
}

TEST(LogicalExpressionRoundTrip, RaggedRuntimeOffsetsSlotBecomesLogicalInputBindingAndRemapsByName) {
    auto expr = std::make_shared<PhysicalExpression>();
    const uint32_t value = appendPhysicalInput(*expr, "value");
    const uint32_t weights = appendPhysicalInput(*expr, "weights");
    const uint32_t offsets_slot = expr->getOrCreateInputSlot("offsets", NamedInput::Kind::Tensor);
    ExprNode offsets{};
    offsets.op = ExprOp::INPUT;
    offsets.input_slot = offsets_slot;
    offsets.input_tensor_dtype = DataType::UINT32;
    expr->nodes.push_back(offsets);

    ExprNode matmul{};
    matmul.op = ExprOp::MATMUL;
    matmul.lhs = value;
    matmul.rhs = weights;
    matmul.matmul_packed_row_binding = MatmulPackedRowBinding::RowsA;
    matmul.matmul_packed_row_capacity = 4096;
    matmul.ragged_runtime_offsets_input_slot = offsets_slot;
    matmul.ragged_runtime_batch_size = 32;
    matmul.ragged_runtime_max_active_values = 4096;
    const uint32_t root = static_cast<uint32_t>(expr->nodes.size());
    expr->nodes.push_back(matmul);

    LogicalExpressionImporter importer(*expr);
    const LogicalExpression logical = importer.importOutputs({{"y", root, {}}}).front().node;
    ASSERT_TRUE(logical->raggedRuntimeOffsetsBinding());
    EXPECT_EQ(logical->raggedRuntimeOffsetsBinding()->name, "offsets");
    EXPECT_EQ(logical->semantics().ragged_runtime_offsets_input_slot, UINT32_MAX);
    EXPECT_NO_THROW(validateLogicalGraph({{"y", logical, {}}}));

    const PhysicalOutputs round_trip = LogicalExpressionLowerer::lower({{"y", logical, {}}});
    const ExprNode& result = round_trip.expr->nodes.at(round_trip.outputs.front().node_idx);
    ASSERT_NE(result.ragged_runtime_offsets_input_slot, UINT32_MAX);
    ASSERT_LT(result.ragged_runtime_offsets_input_slot, round_trip.expr->inputs.size());
    EXPECT_EQ(round_trip.expr->inputs[result.ragged_runtime_offsets_input_slot].name, "offsets");
    EXPECT_EQ(result.matmul_packed_row_capacity, 4096u);
    EXPECT_EQ(result.ragged_runtime_batch_size, 32u);
}

CudaKernelExpression makeRoundTripCudaKernel() {
    return CudaKernelExpression::builder("logical_round_trip")
        .source(R"cuda(
extern "C" __global__ void logical_round_trip(const float* a, const float* b, const float* c, float* out0, float* out1) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    out0[i] = a[i] + b[i] + c[i];
    out1[i] = a[i] - b[i] + c[i];
}
)cuda")
        .entry("logical_round_trip")
        .input("a", DataType::FP32)
        .input("b", DataType::FP32)
        .input("c", DataType::FP32)
        .outputLike("out0", DataType::FP32, "a")
        .outputLike("out1", DataType::FP32, "a")
        .launchGrid1D(CudaKernelExpression::DimExpr::numel("out0"), 64)
        .build();
}

LogicalExpression makeLogicalCudaOutput(const LogicalCudaKernelApplicationPtr& application, uint32_t output_index) {
    ExprNode semantics{};
    semantics.op = ExprOp::CUDA_KERNEL_OUTPUT;
    semantics.cuda_kernel_output_index = output_index;
    semantics.output_dtype = application->specification()->outputs().at(output_index).dtype;
    return LogicalExpressionNode::create(std::move(semantics), {}, std::nullopt, std::nullopt, application);
}

TEST(LogicalExpressionCudaApplicationIdentity, OneApplicationMultipleOutputsShareOneLogicalApplicationAndPhysicalIndex) {
    static_assert(std::is_same_v<LogicalCudaKernelApplicationPtr, std::shared_ptr<const LogicalCudaKernelApplication>>);
    static_assert(!std::is_copy_assignable_v<LogicalCudaKernelApplication>);
    static_assert(!std::is_move_assignable_v<LogicalCudaKernelApplication>);

    auto expr = std::make_shared<PhysicalExpression>();
    const uint32_t shared = appendPhysicalInput(*expr, "shared");
    const uint32_t other = appendPhysicalInput(*expr, "other");
    auto kernel = std::make_shared<CudaKernelExpression>(makeRoundTripCudaKernel());
    expr->cuda_kernel_expressions.push_back(kernel);

    ExprNode out0{};
    out0.op = ExprOp::CUDA_KERNEL_OUTPUT;
    out0.cuda_kernel_spec_index = 0;
    out0.cuda_kernel_output_index = 0;
    out0.output_dtype = DataType::FP32;
    out0.cuda_kernel_input_nodes = {shared, other, shared};
    expr->nodes.push_back(out0);
    const uint32_t out0_index = static_cast<uint32_t>(expr->nodes.size() - 1);

    ExprNode out1 = out0;
    out1.cuda_kernel_output_index = 1;
    expr->nodes.push_back(out1);
    const uint32_t out1_index = static_cast<uint32_t>(expr->nodes.size() - 1);
    const PhysicalOutputs source{.expr = expr, .outputs = {{"out0", out0_index, {}}, {"out1", out1_index, {}}}};

    LogicalExpressionImporter importer(*source.expr);
    const std::vector<LogicalNamedOutput> logical = importer.importOutputs(source.outputs);
    ASSERT_EQ(logical.size(), 2u);
    ASSERT_TRUE(logical[0].node->cudaKernelApplication());
    EXPECT_EQ(logical[0].node->cudaKernelApplication().get(), logical[1].node->cudaKernelApplication().get());
    EXPECT_EQ(logical[0].node->cudaKernelApplication()->specification().get(), kernel.get());
    ASSERT_EQ(logical[0].node->cudaKernelApplication()->inputs().size(), 3u);
    EXPECT_EQ(logical[0].node->cudaKernelApplication()->inputs()[0].get(),
              logical[0].node->cudaKernelApplication()->inputs()[2].get());
    EXPECT_EQ(logical[0].node->cudaKernelApplication()->inputs()[1]->inputBinding()->name, "other");
    EXPECT_NO_THROW(validateLogicalGraph(logical));

    const PhysicalOutputs round_trip = LogicalExpressionLowerer::lower(logical);
    ASSERT_EQ(round_trip.expr->cuda_kernel_expressions.size(), 1u);
    const ExprNode& result0 = round_trip.expr->nodes.at(round_trip.outputs[0].node_idx);
    const ExprNode& result1 = round_trip.expr->nodes.at(round_trip.outputs[1].node_idx);
    ASSERT_EQ(result0.cuda_kernel_input_nodes.size(), 3u);
    EXPECT_EQ(result0.cuda_kernel_input_nodes[0], result0.cuda_kernel_input_nodes[2]);
    EXPECT_EQ(result0.cuda_kernel_input_nodes, result1.cuda_kernel_input_nodes);
    EXPECT_EQ(result0.cuda_kernel_spec_index, result1.cuda_kernel_spec_index);
    EXPECT_EQ(result0.cuda_kernel_output_index, 0u);
    EXPECT_EQ(result1.cuda_kernel_output_index, 1u);
    EXPECT_EQ(round_trip.expr->cuda_kernel_expressions[0]->cacheSignature(), kernel->cacheSignature());
}

TEST(LogicalExpressionCudaApplicationIdentity, SeparatelyAuthoredApplicationsSharingSpecAndInputsLowerToDifferentIndices) {
    auto kernel = std::make_shared<CudaKernelExpression>(makeRoundTripCudaKernel());
    const LogicalExpression shared = makeLogicalInput("shared", NamedInput::Kind::Tensor, DataType::FP32);
    const LogicalExpression other = makeLogicalInput("other", NamedInput::Kind::Tensor, DataType::FP32);
    const std::vector<LogicalExpression> inputs = {shared, other, shared};

    const LogicalCudaKernelApplicationPtr application_a = LogicalCudaKernelApplication::create(kernel, inputs);
    const LogicalCudaKernelApplicationPtr application_b = LogicalCudaKernelApplication::create(kernel, inputs);
    ASSERT_NE(application_a.get(), application_b.get());
    ASSERT_EQ(application_a->specification().get(), application_b->specification().get());

    const LogicalExpression output_a = makeLogicalCudaOutput(application_a, 0);
    const LogicalExpression output_b = makeLogicalCudaOutput(application_b, 0);
    const PhysicalOutputs lowered = LogicalExpressionLowerer::lower({{"a", output_a, {}}, {"b", output_b, {}}});

    ASSERT_EQ(lowered.expr->cuda_kernel_expressions.size(), 2u);
    const ExprNode& physical_a = lowered.expr->nodes.at(lowered.outputs[0].node_idx);
    const ExprNode& physical_b = lowered.expr->nodes.at(lowered.outputs[1].node_idx);
    EXPECT_NE(physical_a.cuda_kernel_spec_index, physical_b.cuda_kernel_spec_index);
    EXPECT_EQ(physical_a.cuda_kernel_input_nodes, physical_b.cuda_kernel_input_nodes);
    EXPECT_EQ(lowered.expr->cuda_kernel_expressions[physical_a.cuda_kernel_spec_index].get(), kernel.get());
    EXPECT_EQ(lowered.expr->cuda_kernel_expressions[physical_b.cuda_kernel_spec_index].get(), kernel.get());
}

TEST(LogicalExpressionCudaApplicationIdentity, ImportDistinguishesPhysicalApplicationIndicesSharingOneSpecPointer) {
    auto expr = std::make_shared<PhysicalExpression>();
    const uint32_t shared = appendPhysicalInput(*expr, "shared");
    const uint32_t other = appendPhysicalInput(*expr, "other");
    auto kernel = std::make_shared<CudaKernelExpression>(makeRoundTripCudaKernel());
    expr->cuda_kernel_expressions = {kernel, kernel};

    ExprNode application_a_output{};
    application_a_output.op = ExprOp::CUDA_KERNEL_OUTPUT;
    application_a_output.cuda_kernel_spec_index = 0;
    application_a_output.cuda_kernel_output_index = 0;
    application_a_output.output_dtype = DataType::FP32;
    application_a_output.cuda_kernel_input_nodes = {shared, other, shared};
    expr->nodes.push_back(application_a_output);
    const uint32_t output_a = static_cast<uint32_t>(expr->nodes.size() - 1);

    ExprNode application_b_output = application_a_output;
    application_b_output.cuda_kernel_spec_index = 1;
    expr->nodes.push_back(application_b_output);
    const uint32_t output_b = static_cast<uint32_t>(expr->nodes.size() - 1);

    LogicalExpressionImporter importer(*expr);
    const auto logical = importer.importOutputs({{"a", output_a, {}}, {"b", output_b, {}}});
    ASSERT_EQ(logical.size(), 2u);
    ASSERT_TRUE(logical[0].node->cudaKernelApplication());
    ASSERT_TRUE(logical[1].node->cudaKernelApplication());
    EXPECT_NE(logical[0].node->cudaKernelApplication().get(), logical[1].node->cudaKernelApplication().get());
    EXPECT_EQ(logical[0].node->cudaKernelApplication()->specification().get(), kernel.get());
    EXPECT_EQ(logical[1].node->cudaKernelApplication()->specification().get(), kernel.get());
    EXPECT_EQ(logical[0].node->cudaKernelApplication()->inputs()[0].get(),
              logical[1].node->cudaKernelApplication()->inputs()[0].get());
    EXPECT_NO_THROW(validateLogicalGraph(logical));
}

TEST(LogicalExpressionCudaApplicationIdentity, PhysicalLogicalPhysicalRoundTripPreservesTwoApplications) {
    auto expr = std::make_shared<PhysicalExpression>();
    const uint32_t shared = appendPhysicalInput(*expr, "shared");
    const uint32_t other = appendPhysicalInput(*expr, "other");
    auto kernel = std::make_shared<CudaKernelExpression>(makeRoundTripCudaKernel());
    expr->cuda_kernel_expressions = {kernel, kernel};

    ExprNode output{};
    output.op = ExprOp::CUDA_KERNEL_OUTPUT;
    output.cuda_kernel_output_index = 0;
    output.output_dtype = DataType::FP32;
    output.cuda_kernel_input_nodes = {shared, other, shared};
    output.cuda_kernel_spec_index = 0;
    expr->nodes.push_back(output);
    const uint32_t first = static_cast<uint32_t>(expr->nodes.size() - 1);
    output.cuda_kernel_spec_index = 1;
    expr->nodes.push_back(output);
    const uint32_t second = static_cast<uint32_t>(expr->nodes.size() - 1);

    LogicalExpressionImporter importer(*expr);
    const auto logical = importer.importOutputs({{"first", first, {}}, {"second", second, {}}});
    const PhysicalOutputs round_trip = LogicalExpressionLowerer::lower(logical);

    ASSERT_EQ(round_trip.expr->cuda_kernel_expressions.size(), 2u);
    const ExprNode& result_first = round_trip.expr->nodes.at(round_trip.outputs[0].node_idx);
    const ExprNode& result_second = round_trip.expr->nodes.at(round_trip.outputs[1].node_idx);
    EXPECT_NE(result_first.cuda_kernel_spec_index, result_second.cuda_kernel_spec_index);
    EXPECT_EQ(result_first.cuda_kernel_input_nodes, result_second.cuda_kernel_input_nodes);
    EXPECT_EQ(round_trip.expr->cuda_kernel_expressions[result_first.cuda_kernel_spec_index].get(), kernel.get());
    EXPECT_EQ(round_trip.expr->cuda_kernel_expressions[result_second.cuda_kernel_spec_index].get(), kernel.get());
}

TEST(LogicalExpressionCudaApplicationIdentity, ImportRejectsOnePhysicalApplicationIndexWithConflictingInputs) {
    auto expr = std::make_shared<PhysicalExpression>();
    const uint32_t a = appendPhysicalInput(*expr, "a");
    const uint32_t b = appendPhysicalInput(*expr, "b");
    auto kernel = std::make_shared<CudaKernelExpression>(makeRoundTripCudaKernel());
    expr->cuda_kernel_expressions.push_back(kernel);

    ExprNode out0{};
    out0.op = ExprOp::CUDA_KERNEL_OUTPUT;
    out0.cuda_kernel_spec_index = 0;
    out0.cuda_kernel_output_index = 0;
    out0.output_dtype = DataType::FP32;
    out0.cuda_kernel_input_nodes = {a, b, a};
    expr->nodes.push_back(out0);
    const uint32_t first = static_cast<uint32_t>(expr->nodes.size() - 1);

    ExprNode out1 = out0;
    out1.cuda_kernel_output_index = 1;
    out1.cuda_kernel_input_nodes = {a, a, b};
    expr->nodes.push_back(out1);
    const uint32_t second = static_cast<uint32_t>(expr->nodes.size() - 1);

    LogicalExpressionImporter importer(*expr);
    EXPECT_THROW((void)importer.importOutputs({{"first", first, {}}, {"second", second, {}}}), std::runtime_error);
}

TEST(LogicalExpressionRoundTrip, RepresentativeNormalizationEmbeddingConditionalAndSegmentedOpsPreserveTopology) {
    const std::vector<ExprOp> ops = {
        ExprOp::LAYERNORM,
        ExprOp::RMSNORM,
        ExprOp::EMBEDDING_LOOKUP,
        ExprOp::WHERE,
        ExprOp::SEGMENTED_SCAN,
        ExprOp::SEGMENTED_REDUCE_SUM,
        ExprOp::SEGMENTED_BROADCAST,
        ExprOp::RAGGED_VALUEWISE_EXTENT,
        ExprOp::RAGGED_CONV1D_CAUSAL,
    };
    for (ExprOp op : ops) {
        SCOPED_TRACE(static_cast<int>(op));
        const PhysicalOutputs source = singleRootPhysicalForOp(op);
        LogicalExpressionImporter importer(*source.expr);
        const auto logical = importer.importOutputs(source.outputs);
        const PhysicalOutputs round_trip = LogicalExpressionLowerer::lower(logical);
        EXPECT_EQ(round_trip.expr->nodes.at(round_trip.outputs.front().node_idx).op, op);
        EXPECT_EQ(canonicalize(round_trip), canonicalize(source));
    }
}


TEST(LogicalExpressionValidation, AcceptsWellFormedSharedGraph) {
    const LogicalExpression x = makeLogicalInput("x");
    const LogicalExpression w = makeLogicalInput("w");
    const LogicalExpression z = makeLogicalBinary(ExprOp::MATMUL, x, w);
    const LogicalExpression a = makeLogicalUnary(ExprOp::SIN, z);
    const LogicalExpression b = makeLogicalUnary(ExprOp::COS, z);

    EXPECT_NO_THROW(validateLogicalGraph({{"a", a, {}}, {"b", b, {}}}));
}

TEST(LogicalExpressionValidation, RejectsDependencyRoleThatDoesNotBelongToOperation) {
    const LogicalExpression x = makeLogicalInput("x");
    ExprNode semantics{};
    semantics.op = ExprOp::SIN;
    const LogicalExpression malformed = LogicalExpressionNode::create(
        semantics,
        {{LogicalDependencyKind::Lhs, 0, x}, {LogicalDependencyKind::Alpha, 0, x}});

    EXPECT_THROW(validateLogicalGraph({{"y", malformed, {}}}), std::runtime_error);
}

TEST(LogicalExpressionValidation, RejectsAttentionFlagWithoutRequiredMetadataDependencies) {
    const LogicalExpression q = makeLogicalInput("q");
    const LogicalExpression k = makeLogicalInput("k");
    const LogicalExpression v = makeLogicalInput("v");
    ExprNode semantics{};
    semantics.op = ExprOp::ATTENTION;
    semantics.attention_use_padding_mask = true;
    const LogicalExpression malformed = LogicalExpressionNode::create(
        semantics,
        {{LogicalDependencyKind::Lhs, 0, q},
         {LogicalDependencyKind::Rhs, 0, k},
         {LogicalDependencyKind::Aux, 0, v}});

    EXPECT_THROW(validateLogicalGraph({{"o", malformed, {}}}), std::runtime_error);
}

TEST(LogicalExpressionRoundTrip, CustomCudaRuntimeScalarAbiKindsSurviveImportAndLowering) {
    auto kernel = std::make_shared<CudaKernelExpression>(
        CudaKernelExpression::builder("logical_scalar_abi_round_trip")
            .source(R"cuda(
extern "C" __global__ void logical_scalar_abi_round_trip(
    const float* x, const float* device_scale, float host_scale, float* y) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    y[i] = x[i] * (*device_scale) * host_scale;
}
)cuda")
            .entry("logical_scalar_abi_round_trip")
            .input("x", DataType::FP32)
            .tensorRuntimeScalarInput("device_scale", DataType::FP32)
            .hostRuntimeScalarInput("host_scale", DataType::FP32)
            .outputLike("y", DataType::FP32, "x")
            .launchGrid1D(CudaKernelExpression::DimExpr::numel("y"), 64)
            .build());

    auto expr = std::make_shared<PhysicalExpression>();
    const uint32_t x = appendPhysicalInput(*expr, "x");
    const uint32_t device_scale = appendPhysicalInput(
        *expr, "device_scale", NamedInput::Kind::TensorRuntimeScalar, DataType::FP32);
    const uint32_t host_scale = appendPhysicalInput(
        *expr, "host_scale", NamedInput::Kind::RuntimeScalarFp32, DataType::FP32);
    expr->cuda_kernel_expressions.push_back(kernel);

    ExprNode output{};
    output.op = ExprOp::CUDA_KERNEL_OUTPUT;
    output.cuda_kernel_spec_index = 0;
    output.cuda_kernel_output_index = 0;
    output.output_dtype = DataType::FP32;
    output.cuda_kernel_input_nodes = {x, device_scale, host_scale};
    const uint32_t root = static_cast<uint32_t>(expr->nodes.size());
    expr->nodes.push_back(output);

    LogicalExpressionImporter importer(*expr);
    const LogicalExpression logical = importer.importOutputs({{"y", root, {}}}).front().node;
    ASSERT_TRUE(logical->cudaKernelApplication());
    ASSERT_EQ(logical->cudaKernelApplication()->inputs().size(), 3u);
    EXPECT_EQ(logical->cudaKernelApplication()->inputs()[0]->op(), ExprOp::INPUT);
    EXPECT_EQ(logical->cudaKernelApplication()->inputs()[1]->op(), ExprOp::TENSOR_RUNTIME_SCALAR);
    EXPECT_EQ(logical->cudaKernelApplication()->inputs()[2]->op(), ExprOp::RUNTIME_SCALAR);
    EXPECT_NO_THROW(validateLogicalGraph({{"y", logical, {}}}));

    const PhysicalOutputs round_trip = LogicalExpressionLowerer::lower({{"y", logical, {}}});
    const ExprNode& result = round_trip.expr->nodes.at(round_trip.outputs.front().node_idx);
    ASSERT_EQ(result.cuda_kernel_input_nodes.size(), 3u);
    EXPECT_EQ(round_trip.expr->nodes[result.cuda_kernel_input_nodes[0]].op, ExprOp::INPUT);
    EXPECT_EQ(round_trip.expr->nodes[result.cuda_kernel_input_nodes[1]].op, ExprOp::TENSOR_RUNTIME_SCALAR);
    EXPECT_EQ(round_trip.expr->nodes[result.cuda_kernel_input_nodes[2]].op, ExprOp::RUNTIME_SCALAR);
}

TEST(LogicalExpressionValidation, RejectsCudaAbiKindMismatch) {
    auto kernel = std::make_shared<CudaKernelExpression>(
        CudaKernelExpression::builder("logical_validation_runtime_scalar")
            .source(R"cuda(
extern "C" __global__ void logical_validation_runtime_scalar(float scale, float* out) {
    if (blockIdx.x == 0 && threadIdx.x == 0) out[0] = scale;
}
)cuda")
            .entry("logical_validation_runtime_scalar")
            .hostRuntimeScalarInput("scale", DataType::FP32)
            .output("out", DataType::FP32, {CudaKernelExpression::DimExpr::constant(1)})
            .launchGrid1D(CudaKernelExpression::DimExpr::constant(1), 1)
            .build());

    const LogicalExpression tensor = makeLogicalInput("not_a_host_scalar", NamedInput::Kind::Tensor, DataType::FP32);
    ExprNode semantics{};
    semantics.op = ExprOp::CUDA_KERNEL_OUTPUT;
    semantics.cuda_kernel_output_index = 0;
    semantics.output_dtype = DataType::FP32;
    const LogicalCudaKernelApplicationPtr application = LogicalCudaKernelApplication::create(kernel, {tensor});
    const LogicalExpression malformed = LogicalExpressionNode::create(
        semantics, {}, std::nullopt, std::nullopt, application);

    EXPECT_THROW(validateLogicalGraph({{"out", malformed, {}}}), std::runtime_error);
}

TEST(LogicalExpressionValidation, ImporterRejectsPhysicalCycles) {
    PhysicalExpression physical;
    ExprNode cyclic{};
    cyclic.op = ExprOp::SIN;
    cyclic.lhs = 0;
    physical.nodes.push_back(cyclic);

    LogicalExpressionImporter importer(physical);
    EXPECT_THROW((void)importer.importOutputs({{"out", 0, {}}}), std::runtime_error);
}

TEST(LogicalExpressionValidation, RejectsDuplicateOrEmptyOutputNames) {
    const LogicalExpression x = makeLogicalInput("x");
    EXPECT_THROW(validateLogicalGraph({{"", x, {}}}), std::runtime_error);
    EXPECT_THROW(validateLogicalGraph({{"same", x, {}}, {"same", x, {}}}), std::runtime_error);
}

}  // namespace
