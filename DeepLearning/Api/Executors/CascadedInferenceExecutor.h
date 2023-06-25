#pragma once

/*
 * This should take a max_batch_size parameter and stamp inference only networks with input sizes:
 * 1, 2, 4, 8, 16, 32, 64, 128 - for max_batch_size = 200, for example.
 * When predict is called, up to 3 stamps will be loaded such that the total batch size is >= the actual batch size,
 * and when it is greater than it is greater by the minimum possible amount when using up to 3 stamps.
 *
 * LocalExecutor's name should be changed to TrainingExecutor
 * local is assumed, for non-local it could be GcpGkeTrainingExecutor or KubernetesTrainingExecutor
 * So there will be two types of executors 1. Training executors 2. inference executors.
 */