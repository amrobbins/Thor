# Custom metric aggregation migration

Custom metrics now have an explicit epoch-aggregation contract and strict batch-validity declarations.

## Breaking API changes

- Every `CustomMetric` must declare `MetricAggregation` with `.aggregation(...)` in C++ or `aggregation=...` in Python.
- `.supportsPartialBatches()` / `supports_partial_batches` was removed. Use `.usesBatchValidity()` / `uses_batch_validity=True` when the expression consumes Thor's batch-validity input.
- Serialized custom metrics must contain both `aggregation` and `uses_batch_validity`. Older architectures without those fields do not load.
- A `RATIO` custom metric expression must emit the public scalar metric plus FP32 scalar outputs named `Thor::METRIC_AGGREGATION_NUMERATOR_NAME` and `Thor::METRIC_AGGREGATION_DENOMINATOR_NAME`.
- The public ratio scalar must equal `denominator == 0 ? 0 : numerator / denominator`, within FP32 rounding tolerance.

The numerator and denominator are internal sufficient statistics. They are not public `NetworkOutput`s and are not serialized as runtime slot or readback state.
