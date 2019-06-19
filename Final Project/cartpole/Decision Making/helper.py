"""Helper functions"""
import random

import numpy as np
import tensorflow as tf


class ExperienceBuffer:
    def __init__(self, buffer_size=10000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[: len(experience) + len(self.buffer) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])


def update_target_graph(tf_vars, tau):
    total_vars = len(tf_vars)
    op_holder = []
    for idx, var in enumerate(tf_vars[: total_vars // 2]):
        op_holder.append(
            tf_vars[idx + total_vars // 2].assign(
                (var.value() * tau)
                + ((1 - tau) * tf_vars[idx + total_vars // 2].value())
            )
        )
    return op_holder


def update_target(op_holder, sess):
    for op in op_holder:
        sess.run(op)
