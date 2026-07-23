# Attention output-residual fusion

This patch set adds a narrow native residual input to `Thor::Attention` and uses it in the SKU patch transformer.

## Apply

From the Thor repository root:

```bash
git apply --check patches/attention-output-residual-fusion-core.patch
git apply patches/attention-output-residual-fusion-core.patch

git apply --check patches/attention-output-residual-fusion-tests.patch
git apply patches/attention-output-residual-fusion-tests.patch
```

From the demand-pipeline repository root, copy or download `demand-pipeline513-transformer-attention-residual-fusion.patch`, then:

```bash
git apply --check demand-pipeline513-transformer-attention-residual-fusion.patch
git apply demand-pipeline513-transformer-attention-residual-fusion.patch
```

## Focused tests

```bash
cd ~/Thor/build
cmake --build .
./thor_tests --gtest_color=yes \
  --gtest_filter='AttentionApi.OutputResidualIsPartOfTheAttentionExpressionAndPreservesPublicShape:AttentionApi.OutputResidualTrainingGraphPlacesForSelfAndCrossAttention'
```

After reinstalling the rebuilt Thor Python extension:

```bash
python -m pytest \
  cart_demand_forecasting/tests/forecasters/test_sku_patch_transformer_forecaster.py \
  -q
```

## Runtime graph change

Before, each transformer attention sublayer was:

```text
Attention output projection -> materialized attention output -> standalone residual add -> state
```

After:

```text
Attention output projection + residual -> state
```

For the default 4-encoder/2-decoder model, this removes eight standalone residual-add layers from the phase-one graph.

The existing FullyConnected epilogues continue to fuse the six FFN residuals and the central history residual. Concatenate pruning is unchanged because Thor already prunes backward paths for concatenations whose inputs do not accept gradients.
