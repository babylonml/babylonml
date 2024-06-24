<b>BabylonML: A Kotlin Deep Learning Framework</b>

This project aims to build a Kotlin deep learning framework that follows a write-once-run-anywhere pattern.

BabylonML offers a high-level DSL (Domain-Specific Language) similar to TensorFlow to quickly build deep learning models and a low-level DSL similar to PyTorch to create custom layers.

The project aims to support running deep learning models on both CPUs using SIMD instructions via the Vector API and on GPUs using the [TornadoVM](https://github.com/beehive-lab/tornadovm) JIT extension. This allows running models on any vendor's GPU, including integrated ones.

Future plans include integration with the [HAT](https://github.com/openjdk/babylon/blob/code-reflection/hat/docs/kernel.md) project API.
