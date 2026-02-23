"""Minimal TensorFlow demo showing the key profiler workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import tensorflow as tf

    from tfmemprof import TFMemoryProfiler

try:
    import tensorflow as tf
except ImportError:
    tf = None

from examples.common import (
    build_simple_tf_model,
    describe_tf_environment,
    generate_tf_batch,
    print_header,
    print_kv,
    print_section,
    run_tf_train_step,
    seed_everything,
)


def profile_tensor_allocation(profiler: TFMemoryProfiler, repeats: int = 3) -> None:
    for idx in range(repeats):

        @profiler.profile_function
        def allocate_batch() -> float:
            inputs, targets = generate_tf_batch(batch_size=128)
            # Return a cheap scalar to keep TF graphs simple
            return float(inputs.mean())

        allocate_batch.__name__ = f"tf_allocate_iter_{idx+1}"
        allocate_batch()


def profile_training_steps(profiler: TFMemoryProfiler, model: tf.keras.Model) -> None:
    for epoch in range(2):
        with profiler.profile_context(f"tf_epoch_{epoch+1}"):
            loss_value = run_tf_train_step(model)
            print_kv(f"Epoch {epoch+1} loss", f"{loss_value:.4f}")


def profile_inference(profiler: TFMemoryProfiler, model: tf.keras.Model) -> None:
    inputs, _ = generate_tf_batch(batch_size=64)
    with profiler.profile_context("tf_inference"):
        logits = model(inputs, training=False)
        probs = tf.nn.softmax(logits)
        entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-9), axis=-1)
        print_kv("Mean entropy", f"{tf.reduce_mean(entropy).numpy():.4f}")


def print_results(results: Any) -> None:
    print_section("Profiler Results")
    print_kv("Duration (s)", f"{results.duration:.3f}")
    print_kv("Peak memory (MB)", f"{results.peak_memory_mb:.2f}")
    print_kv("Average memory (MB)", f"{results.average_memory_mb:.2f}")
    print_kv("Snapshots captured", len(results.snapshots))


def main() -> None:
    seed_everything()
    print_header("GPU Memory Profiler - TensorFlow Demo")

    if tf is None:
        print("TensorFlow is not installed. Skipping TensorFlow demo.")
        return

    from tfmemprof import TFMemoryProfiler

    env = describe_tf_environment()
    print_section("Environment")
    for key, value in env.items():
        print_kv(key, value)

    profiler = TFMemoryProfiler(enable_tensor_tracking=True)

    print_section("Tensor Allocation Profiling")
    profile_tensor_allocation(profiler)

    print_section("Training Loop Profiling")
    model = build_simple_tf_model()
    profile_training_steps(profiler, model)

    print_section("Inference Profiling")
    profile_inference(profiler, model)

    results = profiler.get_results()
    print_results(results)


if __name__ == "__main__":
    main()
