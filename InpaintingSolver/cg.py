import tensorflow as tf

def get_cg(input_shape, steps):
    absolute_residual_stop = 1e-12
    relative_residual_stop = 1e-6

    assert input_shape[-1] == 3
    pixel_count = input_shape[0] * input_shape[1]
    inputs = tf.keras.Input(shape=input_shape)
    image, mask, _ = tf.split(inputs, 3, -1)
    rhs = image * mask
    u = image
    r = rhs - apply_inpainting_operator(u, mask)
    eps0 = eps = tf.reduce_sum(r ** 2, [1, 2, 3], keepdims=True)
    stop = tf.zeros_like(eps, dtype=tf.bool)
    p = r
    for k in range(steps):
        q = apply_inpainting_operator(p, mask)
        pq = tf.reduce_sum(p * q, [1, 2, 3], keepdims=True)
        save_pq = tf.where(pq == 0, 1., pq)
        alpha = tf.where(pq == 0, 0., eps / save_pq)
        stop = stop | (eps / pixel_count < absolute_residual_stop) | (eps / eps0 < relative_residual_stop ** 2)
        u = tf.where(stop, u, u + alpha * p)
        r = tf.where(stop, r, r - alpha * q)
        delta = eps
        eps = tf.reduce_sum(r ** 2, [1, 2, 3], keepdims=True)
        save_delta = tf.where(delta == 0, 1., delta)
        beta = tf.where(delta == 0, 0., eps / save_delta)
        p = r + beta * p
    model = tf.keras.Model(inputs=inputs, outputs=u)
    return model
