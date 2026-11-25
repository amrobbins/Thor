#!/usr/bin/env python3
"""One embedding training step's backward and optimizer update, in PyTorch and FBGEMM, timed the way Thor's
EmbeddingSparseBackwardProfile times its step (graph_step_ms), for a side-by-side comparison on one GPU.

Variants (each given the upstream gradient of an [tokens, dim] lookup into a [vocab, dim] table):
  torch_dense   aten.embedding_dense_backward (the dense [vocab, dim] gradient nn.Embedding makes by default), then
                torch.optim SGD / Adam over the whole table (fused=True where the build has it, else foreach).
  torch_sparse  aten.embedding_sparse_backward (nn.Embedding(sparse=True)'s gradient), then torch.optim.SGD /
                torch.optim.SparseAdam, which update only the rows in the batch.
  fbgemm_tbe    FBGEMM's SplitTableBatchedEmbeddingBagsCodegen, one table, PoolingMode.NONE (a plain embedding),
                EXACT_SGD / ADAM: the optimizer is fused into the backward, as Thor's fused path does. Its forward
                runs before the timed interval (its backward needs it).

Timing: before each step a buffer of twice the L2's size is overwritten and the device synchronized, so a step starts
with a cold L2 on an idle stream; the step is timed with CUDA events from before its first host call to after its last
kernel, so host-side launch work is in the time, as it is in Thor's graph_step_ms. Two input sets alternate.

Indices follow Thor's profiler: "unique" gives every token its own row, "high" gives tokens / 32 rows of 32 tokens
each, in consecutive row ranges. Weights and the upstream gradient are BF16 (FBGEMM: BF16 if the build has it, else
FP16, the same bytes). Optimizer state: PyTorch's Adam keeps its state in the parameter's dtype (BF16); FBGEMM's and
Thor's Adam keep FP32 state, so the PyTorch Adam rows move fewer bytes. Dense Adam also updates every row of the table
each step (its moments decay everywhere); the sparse ones are lazy (only the rows in the batch), as Thor's is.

Every variant is checked once on a fresh table: after one step from zero weights, the touched rows must hold
-lr * G (SGD) or -lr * G / (|G| + eps) (Adam's first step), G the rows' summed gradient, and no other row may change.
"""

import argparse
import csv
import math
import statistics
import sys

import torch

SGD_LR = 0.01
ADAM_LR = 0.001
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vocab", type=int, default=1048576)
    p.add_argument("--dims", default="64,256")
    p.add_argument("--tokens", default="16384,1048576")
    p.add_argument("--dup-modes", default="unique,high")
    p.add_argument("--optimizers", default="sgd,adam")
    p.add_argument("--variants", default="torch_dense,torch_sparse,fbgemm_tbe")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--no-flush", action="store_true", help="do not clear the L2 before each step")
    p.add_argument("--csv", default="", help="also write the rows to this file")
    return p.parse_args()


def int_list(text):
    return [int(x) for x in text.split(",") if x]


def str_list(text):
    return [x for x in text.split(",") if x]


def make_indices(vocab, tokens, mode, slot):
    """Thor's profiler's rows: row = 1 + (base + t % unique - 1) % (vocab - 1)."""
    valid = max(vocab - 1, 1)
    unique = min(tokens, valid)
    if mode == "high":
        unique = max(1, min(valid, max(1, tokens // 32)))
    elif mode != "unique":
        raise ValueError("dup mode " + mode)
    base = 1 + ((slot * unique * 1315423911) % valid)
    t = torch.arange(tokens, device="cuda", dtype=torch.int64)
    return 1 + ((base + (t % unique) - 1) % valid)


def make_upstream(tokens, dim, slot):
    g = torch.Generator(device="cuda")
    g.manual_seed(1000 + slot)
    return (torch.rand(tokens, dim, device="cuda", generator=g) * 2.0 - 1.0).to(torch.bfloat16)


# ---------------------------------------------------------------------------------------------------------------- torch


def torch_optimizer(name, params, sparse):
    """The fastest torch.optim implementation this build offers for the case; returns (optimizer, impl name)."""
    if sparse:
        if name == "sgd":
            return torch.optim.SGD(params, lr=SGD_LR, foreach=False), "SGD"
        return torch.optim.SparseAdam(params, lr=ADAM_LR, betas=ADAM_BETAS, eps=ADAM_EPS), "SparseAdam"
    for impl in ("fused", "foreach"):
        try:
            kwargs = {impl: True}
            if name == "sgd":
                return torch.optim.SGD(params, lr=SGD_LR, **kwargs), "SGD " + impl
            return torch.optim.Adam(params, lr=ADAM_LR, betas=ADAM_BETAS, eps=ADAM_EPS, **kwargs), "Adam " + impl
        except (TypeError, RuntimeError, ValueError):
            continue
    raise RuntimeError("no torch.optim implementation for " + name)


class TorchVariant:
    def __init__(self, sparse, vocab, dim, optimizer):
        self.sparse = sparse
        self.vocab = vocab
        self.weight = torch.nn.Parameter(torch.zeros(vocab, dim, device="cuda", dtype=torch.bfloat16))
        self.opt, self.impl = torch_optimizer(optimizer, [self.weight], sparse)
        self.weights_dtype = "bf16"
        self.state_dtype = "bf16" if optimizer == "adam" else "-"

    def prepare(self, idx, grad):
        return None

    def step(self, idx, grad, prepared):
        if self.sparse:
            g = torch.ops.aten.embedding_sparse_backward(grad, idx, self.vocab, -1, False)
        else:
            g = torch.ops.aten.embedding_dense_backward(grad, idx, self.vocab, -1, False)
        self.weight.grad = g
        self.opt.step()
        self.weight.grad = None

    def weights(self):
        return self.weight.detach()


# --------------------------------------------------------------------------------------------------------------- fbgemm


class TbeVariant:
    def __init__(self, vocab, dim, optimizer, tokens):
        import fbgemm_gpu  # noqa: F401  (registers the operators)
        from fbgemm_gpu.split_embedding_configs import EmbOptimType, SparseType
        from fbgemm_gpu.split_table_batched_embeddings_ops_common import EmbeddingLocation, PoolingMode
        from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
            ComputeDevice,
            SplitTableBatchedEmbeddingBagsCodegen,
        )

        if optimizer == "sgd":
            opt_kwargs = dict(optimizer=EmbOptimType.EXACT_SGD, learning_rate=SGD_LR)
        else:
            opt_kwargs = dict(optimizer=EmbOptimType.ADAM, learning_rate=ADAM_LR, beta1=ADAM_BETAS[0],
                              beta2=ADAM_BETAS[1], eps=ADAM_EPS, weight_decay=0.0)
        specs = [(vocab, dim, EmbeddingLocation.DEVICE, ComputeDevice.CUDA)]
        last_error = None
        self.tbe = None
        for precision_name in ("BF16", "FP16"):
            precision = getattr(SparseType, precision_name, None)
            if precision is None:
                continue
            try:
                self.tbe = SplitTableBatchedEmbeddingBagsCodegen(
                    specs, weights_precision=precision, output_dtype=precision, pooling_mode=PoolingMode.NONE,
                    stochastic_rounding=False, device=torch.device("cuda"), **opt_kwargs)
                self.weights_dtype = precision_name.lower()
                break
            except Exception as e:  # noqa: BLE001  (try the next precision)
                last_error = e
        if self.tbe is None:
            raise RuntimeError("could not build the TBE: " + repr(last_error))
        self.grad_dtype = torch.bfloat16 if self.weights_dtype == "bf16" else torch.float16
        self.impl = "TBE " + opt_kwargs["optimizer"].name
        self.state_dtype = "fp32" if optimizer == "adam" else "-"
        self.offsets = torch.arange(tokens + 1, device="cuda", dtype=torch.int64)
        with torch.no_grad():
            self.tbe.split_embedding_weights()[0].zero_()
        self.grads = {}

    def prepare(self, idx, grad):
        key = grad.data_ptr()
        if key not in self.grads:
            self.grads[key] = grad.to(self.grad_dtype)
        out = self.tbe(idx, self.offsets)
        return out, self.grads[key]

    def step(self, idx, grad, prepared):
        out, g = prepared
        out.backward(g.view(out.shape))

    def weights(self):
        return self.tbe.split_embedding_weights()[0].detach()


def make_variant(name, vocab, dim, optimizer, tokens):
    if name == "torch_dense":
        return TorchVariant(False, vocab, dim, optimizer)
    if name == "torch_sparse":
        return TorchVariant(True, vocab, dim, optimizer)
    if name == "fbgemm_tbe":
        return TbeVariant(vocab, dim, optimizer, tokens)
    raise ValueError("variant " + name)


# ----------------------------------------------------------------------------------------------------------- the check


def check_one_step(variant, optimizer, idx, grad, vocab):
    """One step from zero weights against the expected rows; returns 'ok' or what failed."""
    variant.step(idx, grad, variant.prepare(idx, grad))
    torch.cuda.synchronize()
    w = variant.weights().float()
    touched = torch.unique(idx)
    summed = torch.zeros(vocab, grad.shape[1], device="cuda", dtype=torch.float32)
    summed.index_add_(0, idx, grad.float())
    g = summed[touched]
    if optimizer == "sgd":
        expected = -SGD_LR * g
    else:
        expected = -ADAM_LR * g / (g.abs() + ADAM_EPS)
    got = w[touched]
    err = (got - expected).abs().max().item()
    scale = expected.abs().max().item()
    if not err <= 0.02 * scale + 1e-7:
        return f"rows off by {err:.3g} (scale {scale:.3g})"
    changed_rows = int((w != 0).any(dim=1).sum().item())
    if changed_rows > touched.numel():
        return f"{changed_rows - touched.numel()} rows outside the batch changed"
    del summed, w
    return "ok"


# ----------------------------------------------------------------------------------------------------------- timing


def time_variant(variant, slots, warmup, iters, flush):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for i in range(warmup + iters):
        idx, grad = slots[i % len(slots)]
        prepared = variant.prepare(idx, grad)
        if flush is not None:
            flush.zero_()
        torch.cuda.synchronize()
        start.record()
        variant.step(idx, grad, prepared)
        end.record()
        end.synchronize()
        if i >= warmup:
            times.append(start.elapsed_time(end) * 1000.0)
        del prepared
    times.sort()
    pick = lambda q: times[min(len(times) - 1, int(q * (len(times) - 1) + 0.5))]  # noqa: E731
    return statistics.median(times), pick(0.1), pick(0.9)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        sys.exit("CUDA is not available to this PyTorch")
    props = torch.cuda.get_device_properties(0)
    print(f"# torch {torch.__version__} cuda {torch.version.cuda} device {props.name} "
          f"sm_{props.major}{props.minor} archs {','.join(torch.cuda.get_arch_list())}", file=sys.stderr)
    try:
        import fbgemm_gpu

        print(f"# fbgemm_gpu {getattr(fbgemm_gpu, '__version__', '?')}", file=sys.stderr)
    except Exception as e:  # noqa: BLE001
        print(f"# fbgemm_gpu not importable: {e!r}", file=sys.stderr)

    l2 = getattr(props, "L2_cache_size", 0) or 96 * 1024 * 1024
    flush = None if args.no_flush else torch.empty(2 * l2, dtype=torch.uint8, device="cuda")

    fields = ["variant", "impl", "vocab", "dim", "tokens", "dup_mode", "optimizer", "weights_dtype", "state_dtype",
              "median_us", "p10_us", "p90_us", "check"]
    out = csv.DictWriter(sys.stdout, fieldnames=fields)
    out.writeheader()
    rows = []
    for dim in int_list(args.dims):
        for tokens in int_list(args.tokens):
            for mode in str_list(args.dup_modes):
                slots = [(make_indices(args.vocab, tokens, mode, s), make_upstream(tokens, dim, s)) for s in range(2)]
                for optimizer in str_list(args.optimizers):
                    for name in str_list(args.variants):
                        row = dict(variant=name, impl="", vocab=args.vocab, dim=dim, tokens=tokens, dup_mode=mode,
                                   optimizer=optimizer, weights_dtype="", state_dtype="", median_us="", p10_us="",
                                   p90_us="", check="")
                        print(f"# {name} D{dim} N{tokens} {mode} {optimizer}", file=sys.stderr)
                        try:
                            fresh = make_variant(name, args.vocab, dim, optimizer, tokens)
                            row["check"] = check_one_step(fresh, optimizer, slots[0][0], slots[0][1], args.vocab)
                            del fresh
                            torch.cuda.empty_cache()
                            variant = make_variant(name, args.vocab, dim, optimizer, tokens)
                            row.update(impl=variant.impl, weights_dtype=variant.weights_dtype,
                                       state_dtype=variant.state_dtype)
                            med, p10, p90 = time_variant(variant, slots, args.warmup, args.iters, flush)
                            row.update(median_us=f"{med:.1f}", p10_us=f"{p10:.1f}", p90_us=f"{p90:.1f}")
                            del variant
                        except Exception as e:  # noqa: BLE001  (record it and go on to the next case)
                            row["check"] = "error: " + repr(e).replace("\n", " ")[:300]
                        torch.cuda.empty_cache()
                        out.writerow(row)
                        sys.stdout.flush()
                        rows.append(row)
                del slots
                torch.cuda.empty_cache()
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
