# TrainingProgram / TrainingStep design plan

Thor should keep ordinary supervised training simple while still supporting alternating and multi-objective training patterns such as GANs, actor-critic RL, auxiliary losses, frozen-backbone fine tuning, and self-supervised twin-tower models.

The intended user model is:

```text
Trainer(Network, loader)
  -> interpreted as an implicit single-step TrainingProgram

Trainer(TrainingProgram, loader)
  -> explicit advanced training program
```

A `Network` defines what can be computed. A `TrainingProgram` defines how training proceeds. A `TrainingStep` is one ordered execution/update phase over a network graph.

## Core principles

1. Do not add complexity to the simple path. A network passed directly to the trainer remains the normal one-loss/one-optimizer training surface.
2. Do not add GAN-specific executor behavior. Hinge GAN is one instance of a general alternating optimization pattern.
3. Separate gradient traversal from parameter update. A step may traverse through layers whose parameters are not updated by that step.
4. Keep graph semantics explicit. `StopGradient` is a layer because it is an edge-level gradient barrier, not a trainability flag.
5. Keep trainability and update scoping separate. `freeze_training()` controls default trainability; explicit `TrainingStep.update_parameters` controls advanced per-step updates.
6. Compile each step for performance. The modern trainer should compile each step into a reusable executable plan with its own forward roots, backward roots, parameter-gradient materialization set, and optimizer update set while sharing parameter storage across steps.

## TrainingStep

A training step should eventually own:

```text
name
loss roots
optimizer
update parameter set
input bindings / input providers
requested output tensors / metric tensors
gradient clear / accumulation policy
repeat count or schedule hook
```

The most important invariant is:

```text
backward traversal set != parameter update set
```

For a generator step in a GAN, gradients traverse through the discriminator to reach the generator, but the discriminator parameters are not materialized for update and are not stepped.

## TrainingProgram

The initial `TrainingProgram` is an ordered list of `TrainingStep` objects:

```text
for batch in loader:
  for step in program.steps:
    repeat step.repeat_count times:
      bind step inputs
      run step forward closure
      backprop from step loss roots
      materialize grads only for step update parameters
      run step optimizer update
      publish step outputs/metrics
```

This list model is enough for one-discriminator-step/one-generator-step GANs. A later scheduler can add `every_n_batches`, validation-only steps, EMA update steps, and conditional schedules without changing the core step semantics.

## Hinge GAN shape

```text
D step:
  fake = G(z_d)
  fake = StopGradient(fake)
  real_scores = D(real)
  fake_scores = D(fake)
  loss = HingeGANDiscriminatorLoss(real_scores, fake_scores)
  update_parameters = D.parameters()

G step:
  fake = G(z_g)
  fake_scores = D(fake)
  loss = HingeGANGeneratorLoss(fake_scores)
  update_parameters = G.parameters()
```

The discriminator participates in the generator step forward and backward activation traversal, but its parameters are not update targets for that step.

## Implementation milestones

1. Logical API scaffolding: `ParameterReference`, `TrainingStep`, and `TrainingProgram`.
2. Python bindings and tests for program construction, JSON inspection, parameter reference collection, and duplicate validation.
3. Trainer API accepting either `Network` or `TrainingProgram`, with `Network` converted internally to an implicit single-step program.
4. Compiler support for per-step graph realizations over shared placed parameter storage.
5. Per-step optimizer ownership, parameter-gradient materialization filters, and update sets.
6. Step-local input binding/providers, including random/noise providers for GAN and RL workloads.
7. Optional scheduling combinators such as repeat, validation cadence, and periodic EMA/update-only steps.

This patch starts milestone 1 and the public API/testing portion of milestone 2. It intentionally does not wire the new program object into `LocalExecutor`, because that executor is expected to be replaced.
