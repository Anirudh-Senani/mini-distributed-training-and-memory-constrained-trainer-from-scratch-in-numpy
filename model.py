"""
Mini Distributed Training and Memory-Constrained Trainer from Scratch in NumPy

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - make_synthetic_regression_batch
import numpy as np


def make_synthetic_regression_batch(batch_size, in_dim, out_dim, seed):
    """Return (x, y) where x is (batch_size, in_dim) and y is (batch_size, out_dim) float64."""
    # TODO: seed numpy, sample x, build a hidden teacher, and produce noisy targets y.
    np.random.seed(seed)
    x = np.random.randn(batch_size, in_dim)
    teacher = np.random.randn(in_dim, out_dim)
    y = x @ teacher + np.random.standard_normal((batch_size, out_dim)) * 0.1

    return x, y

# Step 2 - init_mlp_params
def init_mlp_params(in_dim, hidden_dim, out_dim, seed):
    # TODO: return a dict {'W1','b1','W2','b2'} with He-initialized weights and zero biases.
    np.random.seed(seed)
    he_std_w1 = (2/in_dim)**0.5
    he_std_w2 = (2/hidden_dim)**0.5

    W1 = np.random.normal(loc=0.0, scale=he_std_w1, size=(in_dim, hidden_dim))
    b1 = np.zeros((hidden_dim,))
    W2 = np.random.normal(loc=0.0, scale=he_std_w2, size=(hidden_dim, out_dim))
    b2 = np.zeros((out_dim,))

    return dict(
        W1=W1,
        b1=b1,
        W2=W2,
        b2=b2
    )

# Step 3 - linear_forward
def linear_forward(x, w, b):
    # TODO: apply y = x @ w + b and return the resulting (N, out_dim) array
    return x @ w + b

# Step 4 - relu_forward
def relu_forward(x):
    # TODO: apply the ReLU activation elementwise and return an array of the same shape.
    return np.maximum(x, 0.0)

# Step 5 - mlp_forward
def mlp_forward(x, params):
    # TODO: run the two-layer MLP forward and return (y_pred, cache) with keys 'x','z1','a1','z2'.
    z1 = linear_forward(x, params['W1'], params['b1'])
    a1 = relu_forward(z1)
    z2 = linear_forward(a1, params['W2'], params['b2'])

    return z2, dict(x=x, z1=z1, a1=a1, z2=z2)

# Step 6 - mse_loss_and_grad
def mse_loss_and_grad(y_pred, y_true):
    # TODO: compute mean squared error loss and its gradient with respect to y_pred
    dy_pred = (y_pred - y_true)
    loss = (dy_pred**2).mean()

    return float(loss), (2 * dy_pred)/np.prod(y_pred.shape).astype(dy_pred.dtype)

# Step 7 - linear_backward
import numpy as np

def linear_backward(d_out, x, w):
    # TODO: backprop through y = x @ w + b and return (dx, dw, db)
    dx = d_out @ w.T
    dw = x.T @ d_out
    db = d_out.sum(axis=0)

    return dx, dw, db

# Step 8 - relu_backward
def relu_backward(d_out, z):
    # TODO: backprop through ReLU using the pre-activation z, return dz with same shape.
    return np.where(z>0, d_out, 0.0)

# Step 9 - first_linear_backward
def first_linear_backward(d_z1, x, w1):
    # TODO: return gradients (dx, dW1, db1) for z1 = x @ w1 + b1 given d_z1.
    return linear_backward(d_z1, x, w1)

# Step 10 - mlp_backward
def mlp_backward(dy_pred, cache, params):
    # TODO: run the full MLP backward pass returning grads dict with keys W1,b1,W2,b2
    a1, w2, b2 = linear_backward(dy_pred, cache['a1'], params['W2'])
    z1 = relu_backward(a1, cache['z1'])
    x, w1, b1 = first_linear_backward(z1, cache['x'], params['W1'])

    return dict(
        W1=w1,
        b1=b1,
        W2=w2,
        b2=b2
    )

# Step 11 - split_into_micro_batches
def split_into_micro_batches(x, y, micro_batch_size):
    # TODO: split (x, y) into contiguous micro batches of at most micro_batch_size rows.
    batches = []
    for i in range(0, y.shape[0], micro_batch_size):
        batches.append((x[i:i+micro_batch_size], y[i:i+micro_batch_size]))

    return batches

# Step 12 - accumulate_gradients
def accumulate_gradients(accum_grads, new_grads):
    # TODO: return a dict whose values are elementwise sums of accum_grads and new_grads.
    out = {}
    accum_grads = {} if accum_grads is None else accum_grads

    for key in new_grads:
        out[key] = new_grads[key]
        if key in accum_grads and accum_grads[key] is not None:
            out[key] += accum_grads[key]

    return out

# Step 13 - scale_accumulated_gradients
def scale_accumulated_gradients(accum_grads, num_micro_batches):
    # TODO: divide each gradient tensor by num_micro_batches and return a new dict
    out = {}
    for key in accum_grads:
        out[key] = accum_grads[key]/num_micro_batches

    return out

# Step 14 - grad_accumulation_step
def grad_accumulation_step(x, y, params, micro_batch_size):
    # TODO: run forward/backward on each micro batch and combine grads to match a full-batch step.
    accum_grads = None
    micro_batches = split_into_micro_batches(x, y, micro_batch_size)
    N = x.shape[0]
    # num_micro_batches = len(micro_batches)

    for xb, yb in micro_batches:
        y_pred, cache = mlp_forward(xb, params)
        loss, dy_pred = mse_loss_and_grad(y_pred, yb)

        new_grads = mlp_backward(dy_pred, cache, params)
        new_grads = scale_accumulated_gradients(new_grads, 1/xb.shape[0])
        accum_grads = accumulate_gradients(accum_grads, new_grads)

    return scale_accumulated_gradients(accum_grads, N)

# Step 15 - mlp_forward_checkpointed
def mlp_forward_checkpointed(x, params):
    # TODO: forward pass that caches only the block input x, not intermediates.
    z1 = linear_forward(x, params['W1'], params['b1'])
    a1 = relu_forward(z1)
    z2 = linear_forward(a1, params['W2'], params['b2'])

    return z2, dict(x=x)

# Step 16 - recompute_block_activations
def recompute_block_activations(x, params):
    # TODO: recompute z1, a1, z2 from x and params and return them in a cache dict
    z1 = linear_forward(x, params['W1'], params['b1'])
    a1 = relu_forward(z1)
    z2 = linear_forward(a1, params['W2'], params['b2'])

    return dict(x=x, z1=z1, a1=a1, z2=z2)

# Step 17 - mlp_backward_checkpointed
def mlp_backward_checkpointed(dy_pred, light_cache, params):
    # TODO: recompute activations from light_cache['x'] and run the standard MLP backward
    cache = recompute_block_activations(light_cache['x'], params)
    return mlp_backward(dy_pred, cache, params)

# Step 18 - estimate_checkpointing_memory_savings
def estimate_checkpointing_memory_savings(batch_size, in_dim, hidden_dim, out_dim, dtype_bytes):
    # TODO: estimate activation memory in bytes for full vs checkpointed forward on the two-layer MLP.
    full_bytes = ((batch_size*in_dim) + 2 * (batch_size * hidden_dim)) * dtype_bytes
    checkpoint_bytes = (batch_size*in_dim) * dtype_bytes

    return dict(
        full_bytes=full_bytes,
        checkpoint_bytes=checkpoint_bytes,
        saved_bytes=full_bytes-checkpoint_bytes
    )

# Step 19 - cast_to_half_precision
def cast_to_half_precision(values):
    # TODO: Return a new dict mapping each key to its array converted to float16.
    out = {}
    for key in values:
        out[key] = values[key].astype(np.float16)

    return out

# Step 20 - make_master_params
def make_master_params(params):
    # TODO: return a dict mapping the same keys to independent float32 copies of each array.
    out = {}
    for key in params:
        out[key] = params[key].astype(np.float32)

    return out

# Step 21 - scale_loss
def scale_loss(loss, dy_pred, scale):
    # TODO: Scale the scalar loss and the upstream gradient dy_pred by the fixed loss scale.
    return loss*scale, dy_pred*np.array(scale, dtype=dy_pred.dtype)

# Step 22 - unscale_gradients
def unscale_gradients(grads, scale):
    # TODO: divide every gradient tensor by scale and return a new float32 dict
    out = {}
    for key in grads:
        out[key] = (grads[key]/scale).astype(np.float32)

    return out

# Step 23 - has_non_finite_gradients
def has_non_finite_gradients(grads):
    # TODO: return True if any array in grads contains NaN or Inf, else False
    non_finite = False
    for key in grads:
        if not np.all(np.isfinite(grads[key])):
            non_finite = True
            break

    return non_finite

# Step 24 - mixed_precision_step
def mixed_precision_step(x, y, master_params, scale, lr):
    # TODO: run fp16 forward/backward, unscale grads, skip on overflow, else SGD-update fp32 master.
    params = cast_to_half_precision(master_params)
    x_half = x.astype(np.float16)
    y_half = y.astype(np.float16)

    y_pred, light_cache = mlp_forward_checkpointed(x_half, params)
    loss, dy_pred = mse_loss_and_grad(y_pred, y_half)

    loss_scaled, dy_pred_scaled = scale_loss(loss, dy_pred, scale)
    grads_scaled = mlp_backward_checkpointed(dy_pred_scaled, light_cache, params)

    grads = unscale_gradients(grads_scaled, scale)
    if has_non_finite_gradients(grads):
        return loss_scaled, make_master_params(master_params), True

    new_master_params = {}
    for key in master_params:
        new_master_params[key] = master_params[key] - lr * grads[key]

    return loss_scaled, make_master_params(new_master_params), False

# Step 25 - shard_dataset_across_workers
def shard_dataset_across_workers(x, y, num_workers):
    # TODO: split x and y into num_workers contiguous shards along axis 0
    num_samples = [0]*num_workers
    for i in range(x.shape[0]):
        num_samples[i%num_workers] += 1

    shards = []
    ind = 0
    for i in num_samples:
        shards.append((x[ind:ind+i], y[ind:ind+i]))
        ind += i

    return shards

# Step 26 - compute_local_gradients
def compute_local_gradients(x, y, params):
    """Compute parameter gradients for one worker's data shard.

    Forward (mlp_forward) -> loss gradient (mse_loss_and_grad) -> backward
    (mlp_backward). Return a grads dict with keys 'W1', 'b1', 'W2', 'b2'.
    """
    # TODO: forward, then mse loss gradient, then backward; return grads
    y_pred, cache = mlp_forward(x, params)
    # y_pred, light_cache = mlp_forward_checkpointed(x, params)
    
    loss, dy_pred = mse_loss_and_grad(y_pred, y)
    return mlp_backward(dy_pred, cache, params)

# Step 27 - all_reduce_mean
def all_reduce_mean(per_worker_grads):
    # TODO: average a list of gradient dicts elementwise across workers
    grads = {key:per_worker_grads[0][key].copy() for key in per_worker_grads[0]}
    num_workers = len(per_worker_grads)

    for worker_grad in per_worker_grads[1:]:
        for key in grads:
            grads[key] += worker_grad[key]

    for key in grads:
        grads[key] /= num_workers

    return grads

# Step 28 - ring_all_reduce_mean
def ring_all_reduce_mean(per_worker_arrays):
    # TODO: average arrays across workers via ring reduce-scatter then all-gather over chunks.
    N = len(per_worker_arrays)
    shape = per_worker_arrays[0].shape
    size = per_worker_arrays[0].size
    chunk_size = (size + N -1)//N
    out = []
    for arr in per_worker_arrays:
        out_i = np.zeros(N * chunk_size)
        out_i[:size] = arr.ravel()
        out.append(out_i.reshape((N, chunk_size)))

    for s in range(N-1):
        sends = [out[w][(w-s)%N].copy() for w in range(N)]
        for w in range(N):
            out[w][(w-s-1)%N] += sends[(w-1)%N]

    for w in range(N):
        out[w][(w+1)%N] /= N

    for s in range(N-1):
        sends = [out[w][(w+1-s)%N].copy() for w in range(N)]
        for w in range(N):
            out[w][(w-s)%N] = sends[(w-1)%N]

    return out[0].reshape(-1)[:size].reshape(shape)

# Step 29 - data_parallel_train_step
def data_parallel_train_step(x, y, params, num_workers, lr):
    # TODO: shard the batch, compute local gradients, all-reduce mean them, then SGD update params.
    local_grads = [compute_local_gradients(xs, ys, params) for xs, ys in shard_dataset_across_workers(x, y, num_workers)]

    new_params = {}
    for key in local_grads[0].keys():
        new_params[key] = params[key] - lr * ring_all_reduce_mean([grad[key] for grad in local_grads])

    return new_params

# Step 30 - bucket_gradients
def bucket_gradients(grads, bucket_size):
    # TODO: pack flattened gradients into fixed-size buckets and return (buckets, meta).
    bucket_id = 0
    cur_bucket = []
    cur_size = 0
    # cur_meta = []
    buckets = []
    meta = []

    for key in sorted(grads):
        size = grads[key].size
        if cur_size + size > bucket_size:
            buckets.append(cur_bucket)
            # meta.append(cur_meta)
            bucket_id += 1
            cur_size = grads[key].size
            cur_bucket = grads[key].reshape(-1)
            # cur_meta = [(key, grads[key].shape, 0, cur_size, bucket_id)]
            meta.append((key, grads[key].shape, 0, cur_size, bucket_id))

        else:
            # cur_meta.append((key, grads[key].shape, cur_size, cur_size+size, bucket_id))
            meta.append((key, grads[key].shape, cur_size, cur_size+size, bucket_id))
            cur_bucket = np.concatenate([cur_bucket, grads[key].reshape(-1)])
            cur_size += size

    buckets.append(cur_bucket)
    # meta.append(cur_meta)

    return buckets, meta

# Step 31 - init_adam_state
def init_adam_state(params):
    # TODO: build Adam state with zero first/second moments per param and step counter t=0.
    state = {'m':{}, 'v':{}}
    for key in params:
        state['m'][key] = np.zeros_like(params[key])
        state['v'][key] = np.zeros_like(params[key])

    state['t'] = 0

    return state

# Step 32 - partition_optimizer_state
def partition_optimizer_state(state, num_workers):
    # TODO: split each Adam moment tensor into num_workers contiguous flat shards.
    workers = [{'t':state['t'], 'shard_slices':{}, 'm':{}, 'v':{}, 'shapes':{}} for _ in range(num_workers)]

    for key in state['m']:
        size = state['m'][key].size
        chunk_size = (size + num_workers - 1)//num_workers

        for i in range(num_workers):
            workers[i]['shard_slices'][key] = (i*chunk_size, min((i+1)*chunk_size, size))
            workers[i]['m'][key] = state['m'][key].reshape(-1)[i*chunk_size:(i+1)*chunk_size].copy()
            workers[i]['v'][key] = state['v'][key].reshape(-1)[i*chunk_size:(i+1)*chunk_size].copy()
            workers[i]['shapes'][key] = state['m'][key].shape

    return workers

# Step 33 - local_shard_adam_update
def local_shard_adam_update(params, grads, worker_state, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    # TODO: Apply an Adam update to only the local shard of each parameter using its owned moment shards.
    updated_param_shards = {}
    updated_worker_state = {'m':{}, 'v':{}, 't':worker_state['t']+1, 'shard_slices':worker_state['shard_slices'], 'shapes':worker_state['shapes']}

    for key in params:
        start, end = worker_state['shard_slices'][key]
        updated_worker_state['m'][key] = beta1*worker_state['m'][key] + (1-beta1)*grads[key].reshape(-1)[start:end]
        updated_worker_state['v'][key] = beta2*worker_state['v'][key] + (1-beta2)*(grads[key].reshape(-1)[start:end]**2)

        m_hat = updated_worker_state['m'][key]/(1-beta1**updated_worker_state['t'])
        v_hat = updated_worker_state['v'][key]/(1-beta2**updated_worker_state['t'])

        updated_param_shards[key] = params[key].reshape(-1)[start:end] - lr*m_hat/(np.sqrt(v_hat)+eps)

    return updated_param_shards, updated_worker_state

# Step 34 - all_gather_param_shards
def all_gather_param_shards(param_shards_per_worker, shapes, shard_slices_per_worker):
    # TODO: all-gather per-worker 1D parameter shards and restore original shapes.
    updated_params = {}
    num_workers = len(param_shards_per_worker)
    for key in shapes:
        size = np.prod(shapes[key])
        # chunk_size = (size + num_workers - 1)//num_workers
        param = np.zeros(size)
        for i in range(num_workers):
            start, end = shard_slices_per_worker[i][key]
            param[start:end] = param_shards_per_worker[i][key]

        updated_params[key] = param.reshape(shapes[key])

    return updated_params

# Step 35 - zero_optimizer_step
def zero_optimizer_step(params, grads, worker_states, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    # TODO: run a full ZeRO step: each worker updates its shard, then all-gather rebuilds full params
    updated_param_shards, updated_worker_states = [], []
    shapes = worker_states[0]['shapes']
    slices = []

    for worker_state in worker_states:
        ups, ws = local_shard_adam_update(params, grads, worker_state, lr, beta1, beta2, eps)
        updated_param_shards.append(ups)
        updated_worker_states.append(ws)
        slices.append(ws['shard_slices'])

    new_params = all_gather_param_shards(updated_param_shards, shapes, slices)

    return new_params, updated_worker_states

# Step 36 - compute_param_memory_bytes
def compute_param_memory_bytes(params):
    # TODO: sum the total bytes occupied by every parameter array in the dict.
    num_bytes = 0

    for key in params:
        if params[key].dtype.name[-2:].isdigit():
            bytes_per_param = int(params[key].dtype.name[-2:])//8
        else:
            bytes_per_param = int(params[key].dtype.name[-1])//8

        num_bytes += bytes_per_param * params[key].size

    return num_bytes

# Step 37 - compute_optimizer_memory_bytes (not yet solved)
# TODO: implement

# Step 38 - compute_peak_activation_memory_bytes (not yet solved)
# TODO: implement

# Step 39 - compare_memory_with_and_without_optimizations (not yet solved)
# TODO: implement

# Step 40 - full_distributed_training_loop (not yet solved)
# TODO: implement

