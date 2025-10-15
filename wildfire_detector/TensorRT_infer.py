import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np


class TRTInference:
    def __init__(self, engine_path: str):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        # Create execution context (needed to set dynamic shapes at runtime)
        self.context = self.engine.create_execution_context()

        # CUDA stream for asynchronous execution
        self.stream = cuda.Stream()

        # Store tensor names only; buffers will be allocated later in infer()
        self.input_name = None
        self.output_name = None

        # Iterate over all engine tensors (TensorRT 10 API)
        for name in self.engine:
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_name = name

        # Placeholders for device buffers and their sizes
        self.dev_input = None
        self.dev_input_size = 0
        self.dev_output = None
        self.dev_output_size = 0

    def infer(self, np_input: np.ndarray) -> np.ndarray:
        """
        Run inference with TensorRT.
        np_input shape must be (N, C, H, W) float32
        """
        batch_size = np_input.shape[0]

        # 1. Update input shape in context
        self.context.set_input_shape(self.input_name, np_input.shape)

        # 2. Make input contiguous & flatten
        np_input = np.ascontiguousarray(np_input, dtype=np.float32).ravel()

        # 3. (Re)allocate device buffers if needed
        input_size = np_input.nbytes
        if self.dev_input is None or self.dev_input_size < input_size:
            self.dev_input = cuda.mem_alloc(input_size)
            self.dev_input_size = input_size  # update the size tracker
        output_shape = self.context.get_tensor_shape(self.output_name)
        output_size = int(np.prod(output_shape)) * np.float32().nbytes
        if self.dev_output is None or self.dev_output_size < output_size:
            self.dev_output = cuda.mem_alloc(output_size)
            self.dev_output_size = output_size  # update the size tracker

        # 4. Copy input host→device
        cuda.memcpy_htod_async(self.dev_input, np_input, self.stream)

        # 5. Bind addresses
        self.context.set_tensor_address(self.input_name, int(self.dev_input))
        self.context.set_tensor_address(self.output_name, int(self.dev_output))

        # 6. Run inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # 7. Prepare output buffer & copy device→host
        host_output = np.empty(output_shape, dtype=np.float32)
        cuda.memcpy_dtoh_async(host_output, self.dev_output, self.stream)
        self.stream.synchronize()

        # 8. Reshape output to (N, num_classes)
        return host_output.reshape(batch_size, -1)
