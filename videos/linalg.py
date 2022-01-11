import tensorflow as tf


def safe(jitter=1e-6):
    def decorator(fn):
        def new_fn(matrix):
            diag = tf.linalg.diag_part(matrix) + jitter
            B = tf.linalg.set_diag(matrix, diag)
            return fn(B)
        return new_fn
    return decorator


@safe(jitter=1e-8)
def safe_sqrtm(A):
    return tf.linalg.sqrtm(A)


@safe(jitter=1e-8)
def safe_cholesky(A):
    return tf.linalg.cholesky(A)
