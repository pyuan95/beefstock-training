import torch
from torch import nn
from torch import autograd
import cupy as cp
import math


def _find_nearest_divisor(value, target):
    divisors = []
    for i in range(1, value + 1):
        if value % i == 0:
            divisors.append((i, abs(target - i)))
    divisors.sort(key=lambda x: x[1])
    return divisors[0][0]


def _get_best_candidate(optimal, outputs_per_square, num_squares):
    # requirements:
    # 1) num_threads is divisible by num_squares, OR num_squares is divisible by num_threads
    # 2) outputs_per_square * num_squares is disible by num_threads
    best = (float("inf"), None)
    for i in range(1, outputs_per_square * num_squares + 1):
        if (i % num_squares == 0 or num_squares % i == 0) and (outputs_per_square * num_squares) % i == 0:
            best = min(best, (abs(optimal - i), i))
    return best[1]


_num_threads_cache = dict()


def _get_num_threads(outputs_per_square, num_squares):
    optimal_num_threads = 256
    if (outputs_per_square, num_squares) not in _num_threads_cache:
        _num_threads_cache[(outputs_per_square, num_squares)] = _get_best_candidate(optimal_num_threads, outputs_per_square, num_squares)

    return _num_threads_cache[(outputs_per_square, num_squares)]


def _kernel_with_threads(kernel, threads):
    def f(grid, args):
        kernel(grid=grid, block=threads, args=args)

    return f


_feature_transformer_slice_forward_kernel_cache = dict()


def make_feature_transformer_slice_forward_kernel(max_active_features, inputs_per_square, outputs_per_square, num_squares):
    """
    @param: max_active_features
        The maximum number of features that are active
        (non-zero) for a single position. This value determines
        the shape of the inputs.
        This value is of type uint32_t.

    @param: inputs_per_square
        The number of input features per square
        This value is of type uint32.

    @param: outputs_per_square
        The number of outputs per square. Must match the shape of the weights
        This value is of type uint32.

    @param: num_squares
        The number of squares.
        This value is of type uint32.
    """
    num_threads = _get_num_threads(outputs_per_square, num_squares)
    outputs_per_thread = (outputs_per_square * num_squares) // num_threads
    squares_per_thread = max(1, num_squares // num_threads)
    threads_per_square = max(1, num_threads // num_squares)
    key = (
        max_active_features,
        inputs_per_square,
        outputs_per_square,
        num_squares,
        num_threads,
    )
    if key not in _feature_transformer_slice_forward_kernel_cache:
        kernel = cp.RawKernel(
            r"""

typedef unsigned int uint32_t;
typedef int int32_t;

extern "C" __global__
/*
    @assumptions:
        0) The blocks must have dimensionality (BATCH_SIZE,)
        1) num_threads is divisible by num_squares, OR num_squares is divisible by num_threads
        2) outputs_per_square * num_squares is disible by num_threads

    @param: feature_indices
        A matrix of shape (BATCH_SIZE, max_active_features)
        containing indices of active features for each position
        in a batch. Feature index of -1 means that the slot is empty
        and the weights will not be accumulated for it. Moreover
        no further indices from this block will be considered.
        The indices form an implicit matrix of shape
        (BATCH_SIZE, NUM_INPUTS), where the first dimension index is
        inferred from the memory location (BATCH_SIZE), and the
        second dimension index is stored in the feature_indices matrix.
        The type for feature indices is int32_t.

    @param: feature_values
        A matrix of shape (BATCH_SIZE, max_active_features)
        containing the values (arity) of the corresponding
        feature index in feature_indices.
        The type for the feature value (arity) is float32.

    @param: weight
        The weight matrix of shape (inputs_per_square * num_squares, outputs_per_square).
        Weights must be of type float32.

    @param: output
        An output matrix of shape (BATCH_SIZE, outputs_per_square * num_squares).
        It may not be initialized.
        Output values must have type float32.
*/
void feature_transformer_slice_forward(
    const int32_t* const feature_indices,
    const float*   const feature_values,
    const float*   const weight,
          float*   const output
) {{
    float                shared_output[{outputs_per_thread}];
    const uint32_t       block_idx           = blockIdx.x;
    const uint32_t       begin_square        = threadIdx.x / {threads_per_square} * {squares_per_thread};
    const uint32_t       end_square          = begin_square + {squares_per_thread};
    const uint32_t       offset              = (threadIdx.x % {threads_per_square}) * {outputs_per_thread};

    const int32_t* const feature_index_row   = feature_indices + block_idx * {max_active_features};
    const float*   const feature_value_row   = feature_values  + block_idx * {max_active_features};

    for (uint32_t s = 0; s < {outputs_per_thread}; ++s)
    {{
        shared_output[s] = 0.0f;
    }}
    for (uint32_t k = 0; k < {max_active_features}; ++k)
    {{
        const int32_t feature_index = feature_index_row[k];
        const float   feature_value = feature_value_row[k];
        const int32_t sq = feature_index / {inputs_per_square};

        if (feature_index != -1)
        {{
            if (sq >= begin_square && sq < end_square)
            {{
                const float* const weight_slice = weight + feature_index * {outputs_per_square} + offset;
                float* shared_output_slice = shared_output + (sq % {squares_per_thread}) * {outputs_per_square};
                #pragma unroll
                for (uint32_t s = 0; s < ({threads_per_square} > 1 ? {outputs_per_thread} : {outputs_per_square}); ++s)
                {{
                    shared_output_slice[s] += weight_slice[s] * feature_value;
                }}
            }}
        }} else break;
    }}

    float* output_slice = output + block_idx * {output_size} + threadIdx.x * {outputs_per_thread};
    #pragma unroll
    for (uint32_t s = 0; s < {outputs_per_thread}; ++s)
    {{
        output_slice[s] = shared_output[s];
    }}
}}

""".format(
                max_active_features=max_active_features,
                outputs_per_thread=outputs_per_thread,
                threads_per_square=threads_per_square,
                squares_per_thread=squares_per_thread,
                output_size=outputs_per_square * num_squares,
                outputs_per_square=outputs_per_square,
                inputs_per_square=inputs_per_square,
                num_squares=num_squares,
            ),
            "feature_transformer_slice_forward",
        )
        kernel.compile()
        _feature_transformer_slice_forward_kernel_cache[key] = _kernel_with_threads(kernel, (num_threads,))
    return _feature_transformer_slice_forward_kernel_cache[key]


_feature_transformer_slice_backward_kernel_cache = dict()


def make_feature_transformer_slice_backward_kernel(max_active_features, inputs_per_square, outputs_per_square, num_squares):
    """
    @param: max_active_features
        The maximum number of features that are active
        (non-zero) for a single position. This value determines
        the shape of the inputs.
        This value is of type uint32_t.

    @param: inputs_per_square
        The number of input features per square
        This value is of type uint32.

    @param: outputs_per_square
        The number of outputs per square. Must match the shape of the weights
        This value is of type uint32.

    @param: num_squares
        The number of squares.
        This value is of type uint32.
    """
    num_threads = _get_num_threads(outputs_per_square, num_squares)
    outputs_per_thread = (outputs_per_square * num_squares) // num_threads
    squares_per_thread = max(1, num_squares // num_threads)
    threads_per_square = max(1, num_threads // num_squares)
    key = (
        max_active_features,
        inputs_per_square,
        outputs_per_square,
        num_squares,
        num_threads,
    )
    if key not in _feature_transformer_slice_backward_kernel_cache:
        kernel = cp.RawKernel(
            r"""

typedef unsigned int uint32_t;
typedef int int32_t;

extern "C" __global__
/*
    @assumptions:
        0) The blocks must have dimensionality (BATCH_SIZE,)
        1) num_threads is divisible by num_squares, OR num_squares is divisible by num_threads
        2) outputs_per_square * num_squares is disible by num_threads

    @param: feature_indices
        A matrix of shape (BATCH_SIZE, max_active_features)
        containing indices of active features for each position
        in a batch. Feature index of -1 means that the slot is empty
        and the weights will not be accumulated for it. Moreover
        no further indices from this block will be considered.
        The indices form an implicit matrix of shape
        (BATCH_SIZE, NUM_INPUTS), where the first dimension index is
        inferred from the memory location (BATCH_SIZE), and the
        second dimension index is stored in the feature_indices matrix.
        The type for feature indices is int32_t.

    @param: feature_values
        A matrix of shape (BATCH_SIZE, max_active_features)
        containing the values (arity) of the corresponding
        feature index in feature_indices.
        The type for the feature value (arity) is float32.

    @param: weight_grad
        The weight gradient matrix of shape (NUM_INPUTS, output_size).

    @param: output_grad
        An output gradient matrix of shape (BATCH_SIZE, output_size).
        Output values must have type float32.
*/
void feature_transformer_slice_backward(
    const int32_t* const feature_indices,
    const float*   const feature_values,
          float*   const weight_grad,
    const float*   const output_grad
) {{
    float                shared_output_grad[{outputs_per_thread}];
    const uint32_t       block_idx           = blockIdx.x;
    const uint32_t       offset              = (threadIdx.x % {threads_per_square}) * {outputs_per_thread};
    const uint32_t       begin_square        = threadIdx.x / {threads_per_square} * {squares_per_thread};
    const uint32_t       end_square          = begin_square + {squares_per_thread};

    const int32_t* const feature_index_row   = feature_indices + block_idx * {max_active_features};
    const float*   const feature_value_row   = feature_values  + block_idx * {max_active_features};

    const float* output_grad_slice = output_grad + block_idx * {output_size} + threadIdx.x * {outputs_per_thread};
    #pragma unroll
    for (uint32_t s = 0; s < {outputs_per_thread}; ++s)
    {{
        shared_output_grad[s] = output_grad_slice[s];
    }}

    for (uint32_t k = 0; k < {max_active_features}; ++k)
    {{
        const int32_t feature_index = feature_index_row[k];
        const float   feature_value = feature_value_row[k];
        const int32_t sq = feature_index / {inputs_per_square};

        if (feature_index != -1)
        {{
            if (sq >= begin_square && sq < end_square)
            {{
                float* weight_grad_slice = weight_grad + feature_index * {outputs_per_square} + offset;
                float* shared_output_grad_slice = shared_output_grad + (sq % {squares_per_thread}) * {outputs_per_square};
                #pragma unroll
                for (uint32_t s = 0; s < ({threads_per_square} > 1 ? {outputs_per_thread} : {outputs_per_square}); ++s)
                {{
                    const float sog = shared_output_grad_slice[s];
                    if (sog != 0.0f)
                    {{
                        atomicAdd(&weight_grad_slice[s], sog * feature_value);
                    }}
                }}
            }}
        }} else break;
    }}
}}

""".format(
                max_active_features=max_active_features,
                outputs_per_thread=outputs_per_thread,
                threads_per_square=threads_per_square,
                squares_per_thread=squares_per_thread,
                output_size=outputs_per_square * num_squares,
                outputs_per_square=outputs_per_square,
                inputs_per_square=inputs_per_square,
                num_squares=num_squares,
            ),
            "feature_transformer_slice_backward",
        )
        kernel.compile()
        _feature_transformer_slice_backward_kernel_cache[key] = _kernel_with_threads(kernel, (num_threads,))
    return _feature_transformer_slice_backward_kernel_cache[key]


class FeatureTransformerSliceFunction(autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inputs_per_square,
        outputs_per_square,
        num_squares,
        feature_indices,
        feature_values,
        weight,
    ):
        ctx.inputs_per_square = inputs_per_square
        ctx.outputs_per_square = outputs_per_square
        ctx.num_squares = num_squares
        ctx.save_for_backward(
            feature_indices,
            feature_values,
            weight,
        )
        assert len(feature_indices.shape) == 2
        assert len(feature_values.shape) == 2
        assert feature_indices.shape[0] == feature_values.shape[0]
        assert feature_indices.shape[1] == feature_values.shape[1]
        assert feature_indices.dtype == torch.int32
        assert feature_values.dtype == torch.float32

        assert len(weight.shape) == 2
        assert weight.dtype == torch.float32

        assert feature_indices.is_cuda
        assert feature_values.is_cuda
        assert weight.is_cuda

        assert feature_values.device == feature_indices.device
        assert weight.device == feature_indices.device

        assert feature_indices.is_contiguous()
        assert feature_values.is_contiguous()
        assert weight.is_contiguous()

        assert outputs_per_square == weight.shape[1]

        device = feature_indices.device
        batch_size = feature_indices.shape[0]
        max_active_features = feature_indices.shape[1]
        output_size = outputs_per_square * num_squares

        output = torch.zeros(
            batch_size,
            output_size,
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )

        kernel = make_feature_transformer_slice_forward_kernel(max_active_features, inputs_per_square, outputs_per_square, num_squares)
        kernel(
            grid=(batch_size,),
            args=(
                feature_indices.data_ptr(),
                feature_values.data_ptr(),
                weight.data_ptr(),
                output.data_ptr(),
            ),
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        assert not ctx.needs_input_grad[0]
        assert not ctx.needs_input_grad[1]
        assert not grad_output.isnan().any()

        inputs_per_square = ctx.inputs_per_square
        outputs_per_square = ctx.outputs_per_square
        num_squares = ctx.num_squares

        grad_output = grad_output.contiguous()

        feature_indices, feature_values, weight = ctx.saved_tensors

        device = feature_indices.device
        batch_size = feature_indices.shape[0]
        max_active_features = feature_indices.shape[1]

        weight_grad = torch.zeros(weight.shape[0], weight.shape[1], dtype=torch.float32, device=device)

        kernel = make_feature_transformer_slice_backward_kernel(max_active_features, inputs_per_square, outputs_per_square, num_squares)
        kernel(
            grid=(batch_size,),
            args=(
                feature_indices.data_ptr(),
                feature_values.data_ptr(),
                weight_grad.data_ptr(),
                grad_output.data_ptr(),
            ),
        )
        assert not weight_grad.isnan().any()
        return None, None, None, None, None, weight_grad


class FeatureTransformerSlice(nn.Module):
    def __init__(self, inputs_per_square, outputs_per_square, num_squares):
        super(FeatureTransformerSlice, self).__init__()
        self.inputs_per_square = inputs_per_square
        self.outputs_per_square = outputs_per_square
        self.num_squares = num_squares

        sigma = math.sqrt(1 / (outputs_per_square * num_squares))
        temp = torch.rand(inputs_per_square, outputs_per_square, dtype=torch.float32) * sigma * 2 - sigma
        temp = temp.reshape(1, inputs_per_square, outputs_per_square) + torch.zeros(num_squares, 1, 1)
        self.weight = nn.Parameter(temp.reshape(-1, outputs_per_square))

    def forward(self, feature_indices, feature_values):
        return FeatureTransformerSliceFunction.apply(
            self.inputs_per_square,
            self.outputs_per_square,
            self.num_squares,
            feature_indices,
            feature_values,
            self.weight,
        )


if __name__ == "__main__":

    def FeatureTransformerSliceFunctionEmulate(
        feature_indices,
        feature_values,
        num_squares,
        inputs_per_square,
        outputs_per_square,
        weight,
    ):
        batch_size = feature_indices.shape[0]
        result = torch.zeros([batch_size, num_squares * outputs_per_square], dtype=torch.float32)
        device = weight.device
        feature_indices = feature_indices.cpu()
        feature_values = feature_values.cpu()
        weight = weight.cpu()
        for i in range(feature_indices.shape[0]):
            for j in range(feature_indices.shape[1]):
                feature_index = feature_indices[i, j]
                if feature_index >= 0:
                    value = feature_values[i, j]
                    sq = feature_index // inputs_per_square
                    result[i, sq * outputs_per_square : (sq + 1) * outputs_per_square] += weight[feature_index] * value

        return result.to(device)

    def test():
        BATCH_SIZE = 64
        NUM_SQUARES = 64
        MAX_ACTIVE_FEATURES = 300
        INPUTS_PER_SQUARE = 64
        OUTPUTS_PER_SQUARE = 128
        NUM_INPUTS = NUM_SQUARES * INPUTS_PER_SQUARE
        MAX_ERROR = 1e-4

        torch.manual_seed(0)
        weight0 = torch.rand(
            NUM_INPUTS,
            OUTPUTS_PER_SQUARE,
            dtype=torch.float32,
            requires_grad=True,
        )
        torch.manual_seed(0)
        weight1 = torch.rand(
            NUM_INPUTS,
            OUTPUTS_PER_SQUARE,
            dtype=torch.float32,
            requires_grad=True,
        )
        indices = (torch.rand(BATCH_SIZE, MAX_ACTIVE_FEATURES) * (NUM_INPUTS + 1) - 1).to(dtype=torch.int32)
        values = torch.rand(BATCH_SIZE, MAX_ACTIVE_FEATURES, dtype=torch.float32)
        output0 = FeatureTransformerSliceFunctionEmulate(
            indices.clone(),
            values.clone(),
            NUM_SQUARES,
            INPUTS_PER_SQUARE,
            OUTPUTS_PER_SQUARE,
            weight0,
        )
        output1 = FeatureTransformerSliceFunction.apply(
            INPUTS_PER_SQUARE,
            OUTPUTS_PER_SQUARE,
            NUM_SQUARES,
            indices.cuda(),
            values.cuda(),
            weight1.cuda(),
        )
        assert torch.max(output0.cpu() - output1.cpu()) < MAX_ERROR
        output0.cpu().sum().backward()
        output1.cpu().sum().backward()
        assert torch.max(weight0.grad.cpu() - weight1.grad.cpu()) < MAX_ERROR
        print("Tests passed.")

    test()
