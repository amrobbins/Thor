#pragma once

/**
 * Logical dataset partition selected for batch iteration.
 *
 * The numeric values are retained for compatibility with the historical shard
 * storage format. Dataset ownership of partition membership lives in
 * DatasetSplitManifest; this enum only selects a partition at execution time.
 */
enum class ExampleType { TRAIN = 3, VALIDATE, TEST };
