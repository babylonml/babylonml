package com.babylonml.backend.inference.operations.tornadovm

import com.babylonml.backend.inference.tornadovm.InputSource
import uk.ac.manchester.tornado.api.types.tensors.Tensor

class TensorInputSource(private val data: Tensor) : InputSource
