"""
implementation of centered Adam (cAdam) optimizer, a variation of Adam.

Original Adam code modified by Sebastian Jost to create cAdam (using m_t as gradient approximation).
"""

import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops

class cAdam(keras.optimizers.Optimizer):
    def __init__(self,
                 learning_rate=0.01,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-7,
                 name="cAdam",
                 **kwargs):
        """Call super().__init__() and use _set_hyper() to store hyperparameters"""
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate)) # handle lr=learning_rate
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
            self.add_slot(var, "m_t") # gradient estimation for each component
        for var in var_list:
            self.add_slot(var, "v_t") # variance estimation for each component


    def _prepare_local(self, var_device, var_dtype, apply_state):
        """
        ? prepare variables for one optimization step that stay the same for all weights
        """
        local_step = math_ops.cast(self.iterations + 1, var_dtype)
        beta_1 = array_ops.identity(self._get_hyper('beta_1', var_dtype))
        beta_2 = array_ops.identity(self._get_hyper('beta_2', var_dtype))

        beta_1_power = math_ops.pow(beta_1, local_step)
        beta_2_power = math_ops.pow(beta_2, local_step)

        # lr = apply_state[(var_device, var_dtype)]['learning_rate']
        # lr = self._get_hyper("learning_rate")
        # use definition 1 of Adam: v_t_hat
        lr = self._get_hyper("learning_rate") #* math_ops.sqrt(1 - beta_2_power)#/ (1 - beta_1_power)

        one_minus_beta_1 = self._get_hyper("one_minus_beta_1")
        one_minus_beta_2 = self._get_hyper("one_minus_beta_2")
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

    # @tf.function
    def _resource_apply_dense(self, grad, var, apply_state=None):
        """
        Update the slots and perform one optimization step for one model variable

        m_t = grad * (1 - beta_1) + m_t_minus_1 * beta_1
        m_t_hat = m_t / (1 - beta_1_power)

        centered_grad = grad - m_t
        v_t = centered_grad**2 * (1 - beta_2) + v_t_minus_1 * beta_2
        v_t_hat = v_t / sqrt(1 - beta_2_power)

        new_theta = theta + lr * m_t_hat / (sqrt(v_t_hat) + epsilon)
        """
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))
        # beta_1_power = self._get_hyper("beta_1_power")
        # beta_2_power = self._get_hyper("beta_2_power")
        # epsilon = self._get_hyper("epsilon")
        # beta_1 = self._get_hyper("beta_1")
        # beta_2 = self._get_hyper("beta_2")
        # one_minus_beta_1 = self._get_hyper("one_minus_beta_1")
        # one_minus_beta_2 = self._get_hyper("one_minus_beta_2")
        # lr = self._get_hyper("learning_rate")

        # # update beta power variables
        # beta_1_power = beta_1_power.assign(beta_1_power * beta_1)
        # beta_2_power = beta_2_power.assign(beta_2_power * beta_2)
        # load and update m_t
        m_tm1 = self.get_slot(var, "m_t")
        m_t = state_ops.assign(m_tm1,
            grad * coefficients["one_minus_beta_1"] + m_tm1 * coefficients["beta_1"])
        m_t_hat = m_t / (1 - coefficients["beta_1_power"])

        # load and update v_t
        v_tm1 = self.get_slot(var, "v_t")
        centered_v = grad - m_t
        # tf.print(math_ops.reduce_euclidean_norm((centered_v)**2))
        v_t = state_ops.assign(v_tm1,
            (centered_v * centered_v) * coefficients["one_minus_beta_2"] + v_tm1 * coefficients["beta_2"])

        # v_sqrt = math_ops.sqrt(v_t)
        # use definition 1 of Adam: v_t_hat
        v_t_hat = v_t / (1 - coefficients["beta_2_power"])
        v_sqrt = math_ops.sqrt(v_t_hat)

        # update parameter values
        var_update = state_ops.assign_sub(var,
                coefficients["lr"] * m_t_hat/(v_sqrt + coefficients["epsilon"])
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