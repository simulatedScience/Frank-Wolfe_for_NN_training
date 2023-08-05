"""
This module reimplements the optimizer Adam to test the speed of the custom optimizer implementations.

Original Adam code from keras library
"""

import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops


class my_Adam(keras.optimizers.Optimizer):
    def __init__(self,
                 learning_rate=0.01,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-7,
                 name="my_adam",
                 **kwargs):
        """Call super().__init__() and use _set_hyper() to store hyperparameters"""
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get(
            "lr", learning_rate))  # handle lr=learning_rate
        self._set_hyper("beta_1", beta_1)
        self._set_hyper("beta_2", beta_2)
        self._set_hyper("beta_1_power", beta_1)
        self._set_hyper("beta_2_power", beta_2)
        self._set_hyper("one_minus_beta_1", 1 - beta_1)
        self._set_hyper("one_minus_beta_2", 1 - beta_2)
        self._set_hyper("epsilon", epsilon)

    def _create_slots(self, var_list):
        """
        For each model variable, create the optimizer variable associated with it.
        TensorFlow calls these optimizer variables "slots".
        For momentum optimization, we need one momentum slot per model variable.
        """
        for var in var_list:
            self.add_slot(var, "m_t")  # gradient estimation for each component
        for var in var_list:
            self.add_slot(var, "v_t")  # variance estimation for each component

    def _prepare_local(self, var_device, var_dtype, apply_state):
        """
        ? prepare variables for one optimization step that stay the same for all weights
        """
        local_step = math_ops.cast(self.iterations + 1, var_dtype)
        beta_1 = array_ops.identity(self._get_hyper('beta_1', var_dtype))
        beta_2 = array_ops.identity(self._get_hyper('beta_2', var_dtype))

        # update beta powers beta_1^t and beta_2^t
        beta_1_power = math_ops.pow(beta_1, local_step)
        beta_2_power = math_ops.pow(beta_2, local_step)

        # use definition 1 of Adam: v_t_hat
        lr = self._get_hyper("learning_rate") / \
            (1 - beta_1_power) * math_ops.sqrt(1 - beta_2_power)

        one_minus_beta_1 = self._get_hyper("one_minus_beta_1")
        one_minus_beta_2 = self._get_hyper("one_minus_beta_2")
        # update settings for the next optimization step
        apply_state[(var_device, var_dtype)].update(
            dict(
                lr=lr,
                epsilon=ops.convert_to_tensor_v2_with_dispatch(
                    self.epsilon, var_dtype),
                beta_1=beta_1,
                beta_1_power=beta_1_power,
                one_minus_beta_1=one_minus_beta_1,
                beta_2=beta_2,
                beta_2_power=beta_2_power,
                one_minus_beta_2=one_minus_beta_2))

    def _resource_apply_dense(self, grad, var, apply_state=None):
        """
        Update the slots and perform one optimization step for one model variable

        lr_t = lr * sqrt(1 - beta_2_power) / (1 - beta_1_power)

        m_t = grad * (1 - beta_1) + m_t_minus_1 * beta_1

        centered_grad = grad - m_t
        v_t = centered_grad**2 * (1 - beta_2) + v_t_minus_1 * beta_2

        new_theta = theta + lr_t * m_t / (sqrt(v_t) + epsilon)
        """
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))
        # load and update m_t
        m_tm1 = self.get_slot(var, "m_t")
        m_t = state_ops.assign(m_tm1,
                               grad * coefficients["one_minus_beta_1"] + m_tm1 * coefficients["beta_1"])
        # m_t_hat = m_t / (1 - coefficients["beta_1_power"])

        # load and update v_t
        v_tm1 = self.get_slot(var, "v_t")
        # tf.print(math_ops.reduce_euclidean_norm((centered_v)**2))
        v_t = state_ops.assign(v_tm1,
                               (grad * grad) * coefficients["one_minus_beta_2"] + v_tm1 * coefficients["beta_2"])

        v_sqrt = math_ops.sqrt(v_t)

        # update parameter values
        var_update = state_ops.assign_sub(var,
                                          coefficients["lr"] * m_t /
                                          (v_sqrt + coefficients["epsilon"])
                                          )
        return control_flow_ops.group(*[var_update, m_t, v_t])

    def _resource_apply_sparse(self, grad, var):
        """
        apply optimization step with sparse gradients.
        (not implemented for cAdam)
        """
        raise NotImplementedError("Sparse updates not supported.")

    def get_config(self):
        """
        return configuration of the optimizer.
        """
        base_config = super().get_config()
        return {
            **base_config,
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "beta_1": self._serialize_hyperparameter("beta_1"),
            "beta_2": self._serialize_hyperparameter("beta_2"),
            "epsilon": self._serialize_hyperparameter("epsilon"),
        }
