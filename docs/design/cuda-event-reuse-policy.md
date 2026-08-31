# CUDA event reuse and dependency ownership

Thor treats a CUDA event used for recurring stream synchronization as part of
the logical dependency edge, not as a disposable token and not as a property of
a stream pair.

## Fixed recurring dependency edges

An execution object that repeatedly publishes the same dependency owns a
persistent `Event`:

```cpp
Event producerReadyEvent;

producer.putEvent(producerReadyEvent);
consumer.waitEvent(producerReadyEvent);
```

or, equivalently:

```cpp
consumer.waitFor(producer, producerReadyEvent);
```

`Stream::putEvent(Event&)` lazily creates the CUDA event once and subsequently
re-records the same `cudaEvent_t`. The event's GPU, timing-enabled flag, and
blocking-host-synchronization flag are immutable and must match on every reuse.

The value-returning `Stream::putEvent()` remains appropriate when a distinct
completion token is intentionally created and ownership escapes to another
object or caller. It should not be the default for a steady-state recurring
stream-to-stream dependency.

Thor deliberately does not keep a hidden `(producer stream, consumer stream)`
event cache. Multiple independent logical dependencies may use the same stream
pair, and hidden sharing would make their record/wait ordering interfere.

## Dynamic submission-local dependency graphs

When the number of dependency edges is known only for a particular submission,
use `ReusableEventPool` / `ReusableEventLeases` rather than permanent object
members. The shared facility is thread-local to avoid a global lease lock and
separates free events by GPU, timing intent, and blocking-sync intent.

An event may be returned to the pool after **all** `cudaEventRecord` and
`cudaStreamWaitEvent` API calls for its current logical use have been issued.
CUDA defines an already-enqueued stream wait against the event generation that
was current when the wait call was made; a subsequent `cudaEventRecord` does not
retarget that earlier wait. This is the property that makes post-submission
re-recording safe.

Do not return an event to the pool while code can still enqueue a wait intended
for its current generation.

## Steady-state invariant

The target policy is:

> A recurring stream-to-stream dependency should not create and destroy a CUDA
> event on every execution.

This does not imply that every value-returning `putEvent()` is wrong. Timing
measurements, host-visible completion tokens, asynchronous ownership transfer,
and genuinely distinct outstanding completion events may require separate CUDA
events.

## Final audit rule

The production-source audit distinguishes recurring dependency events from
escaping completion tokens. Active stream-to-stream synchronization must not
use `consumer.waitEvent(producer.putEvent(...))`; fixed edges own persistent
`Event` storage and dynamic submission graphs lease reusable events.

Value-returning `putEvent()` remains intentionally allowed where the returned
event is a distinct token whose ownership escapes the recording scope. Current
examples include optimizer completion tokens, stamp/batch completion tokens
when the caller does not provide reusable storage, host synchronization
snapshots, asynchronous archive/data-session completion tokens, and cuBLAS
autotuning timing events. These sites must not be mechanically converted to a
single persistent event because multiple generations may be outstanding.
