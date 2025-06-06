#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Gradients for soft min force."""

from tensorflow.python.framework import (
    ops,
)

from deepmd.tf.env import (
    op_grads_module,
)


@ops.RegisterGradient("SoftMinForce")
def _soft_min_force_grad_cc(op, grad):
    net_grad = op_grads_module.soft_min_force_grad(
        grad,
        op.inputs[0],
        op.inputs[1],
        op.inputs[2],
        op.inputs[3],
        n_a_sel=op.get_attr("n_a_sel"),
        n_r_sel=op.get_attr("n_r_sel"),
    )
    return [net_grad, None, None, None]
