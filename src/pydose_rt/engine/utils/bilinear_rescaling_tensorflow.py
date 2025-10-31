import tensorflow as tf
from typing import Tuple


def bilinear_rescaling_matrix(
    img_size: Tuple[float, float],
    center: Tuple[float, float],
    scale_factor: float
) -> tf.sparse.SparseTensor:
    """
    Generate a sparse transformation matrix for bilinear rescaling of an image in TensorFlow.

    Parameters:
    -----------
    img_size : Tuple[int, int]
        Image size as (height, width).

    center : Tuple[float, float]
        Center point (y, x) for scaling.

    scale_factor : float
        Scaling factor >1 zooms in, <1 zooms out.

    Returns:
    --------
    tf.sparse.SparseTensor
        Sparse (H*W, H*W) bilinear rescaling matrix.
    """
    H, W = img_size
    H = int(H)
    W = int(W)

    # Create target grid
    y_tgt, x_tgt = tf.meshgrid(
        tf.range(H, dtype=tf.float32),
        tf.range(W, dtype=tf.float32),
        indexing='ij'
    )

    # Map target pixel locations to source coords
    y_src = (y_tgt - center[0]) / scale_factor + center[0]
    x_src = (x_tgt - center[1]) / scale_factor + center[1]

    y0 = tf.floor(y_src)
    x0 = tf.floor(x_src)
    y1 = y0 + 1
    x1 = x0 + 1

    y0_int = tf.cast(y0, tf.int32)
    x0_int = tf.cast(x0, tf.int32)
    y1_int = tf.cast(y1, tf.int32)
    x1_int = tf.cast(x1, tf.int32)

    dy = tf.clip_by_value(y_src - y0, 0.0, 1.0)
    dx = tf.clip_by_value(x_src - x0, 0.0, 1.0)

    weights = [
        (1 - dy) * (1 - dx),  # top-left
        (1 - dy) * dx,        # top-right
        dy * (1 - dx),        # bottom-left
        dy * dx               # bottom-right
    ]
    neighbor_offsets = [
        (y0_int, x0_int),
        (y0_int, x1_int),
        (y1_int, x0_int),
        (y1_int, x1_int)
    ]

    tgt_indices = tf.reshape(tf.range(H * W, dtype=tf.int32), (H, W))

    all_rows = []
    all_cols = []
    all_vals = []

    for (yn, xn), w in zip(neighbor_offsets, weights):
        valid = tf.where(
            (yn >= 0) & (yn < H) & (xn >= 0) & (xn < W)
        )
        yn_valid = tf.gather_nd(yn, valid)
        xn_valid = tf.gather_nd(xn, valid)
        w_valid = tf.gather_nd(w, valid)

        src_idx = yn_valid * W + xn_valid
        tgt_idx = tf.gather_nd(tgt_indices, valid)

        all_rows.append(tgt_idx)
        all_cols.append(src_idx)
        all_vals.append(w_valid)

    rows = tf.concat(all_rows, axis=0)
    cols = tf.concat(all_cols, axis=0)
    vals = tf.concat(all_vals, axis=0)

    indices = tf.stack([rows, cols], axis=1)
    sparse_mat = tf.sparse.SparseTensor(indices=tf.cast(indices, dtype=tf.int64), values=vals, dense_shape=[H * W, H * W])
    sparse_mat = tf.sparse.reorder(sparse_mat)
    sparse_mat = tf.sparse.transpose(sparse_mat)  # Shape: (HW, HW)
    return sparse_mat

def apply_transformation_matrix(image: tf.Tensor, matrix: tf.sparse.SparseTensor) -> tf.Tensor:
    """
    Efficient application of sparse transformation matrix to (C, H, W) image batch.

    Args:
        image: tf.Tensor of shape (C, H, W)
        matrix: tf.sparse.SparseTensor of shape (HW, HW)

    Returns:
        Transformed tf.Tensor of shape (C, H, W)
    """
    C = tf.shape(image)[0]
    H = tf.shape(image)[1]
    W = tf.shape(image)[2]

    image_flat = tf.reshape(image, (C, H * W))

    def single_channel_fn(vec):
        # vec shape: (HW,)
        return tf.sparse.sparse_dense_matmul(matrix, tf.expand_dims(vec, axis=-1))[:, 0]

    result_flat = tf.vectorized_map(single_channel_fn, image_flat)  # Shape: (C, HW)
    return tf.reshape(result_flat, (C, H, W))

def apply_transformation_matrix(image: tf.Tensor, matrix: tf.sparse.SparseTensor) -> tf.Tensor:
    """
    Apply a sparse transformation matrix to a (C, H, W) image or batch of grayscale images.

    Args:
        image: Tensor of shape (C, H, W) or (C, H, W, 1)
        matrix: tf.sparse.SparseTensor of shape (H*W, H*W)

    Returns:
        Transformed image of shape (C, H, W)
    """
    if image.shape.rank == 4:
        image = tf.squeeze(image, axis=-1)  # Remove the trailing singleton dimension if present

    C = tf.shape(image)[0]
    H = tf.shape(image)[1]
    W = tf.shape(image)[2]

    image_flat = tf.reshape(image, (C, H * W))  # Shape: (C, HW)

    # Now we can multiply directly: (C, HW) @ (HW, HW)^T = (C, HW)
    result_flat = tf.sparse.sparse_dense_matmul(image_flat, matrix)

    return tf.reshape(result_flat, (C, H, W))
