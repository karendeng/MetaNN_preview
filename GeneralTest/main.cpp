#include "policies/test_change_policy.h"
#include "policies/test_policy_operations.h"
#include "policies/test_policy_selector.h"
#include "facilities/test_named_params.h"
#include "evaluate/test_eval_plan.h"
#include "data/test_cpu_matrix.h"
#include "data/test_one_hot_vector.h"
#include "data/test_trival_matrix.h"
#include "data/test_zero_matrix.h"
#include "operators/test_abs.h"
#include "operators/test_add.h"
#include "operators/test_collapse.h"
#include "operators/test_divide.h"
#include "operators/test_dot.h"
#include "operators/test_element_mul.h"
#include "operators/test_interpolate.h"
#include "operators/test_negative_log_likelihood.h"
#include "operators/test_negative_log_likelihood_derivative.h"
#include "operators/test_sigmoid.h"
#include "operators/test_sigmoid_derivative.h"
#include "operators/test_sign.h"
#include "operators/test_softmax.h"
#include "operators/test_softmax_derivative.h"
#include "operators/test_substract.h"
#include "operators/test_tanh.h"
#include "operators/test_tanh_derivative.h"
#include "operators/test_transpose.h"
#include "layers/elementary/test_abs_layer.h"
#include "layers/elementary/test_add_layer.h"
#include "layers/elementary/test_bias_layer.h"
#include "layers/elementary/test_element_mul_layer.h"
#include "layers/elementary/test_interpolate_layer.h"
#include "layers/elementary/test_sigmoid_layer.h"
#include "layers/elementary/test_softmax_layer.h"
#include "layers/elementary/test_tanh_layer.h"
#include "layers/elementary/test_weight_layer.h"
#include "layers/cost/test_negative_log_likelihood_layer.h"
#include "layers/compose/test_compose_kernel.h"
#include "layers/compose/test_linear_layer.h"
#include "layers/compose/test_single_layer.h"
#include "layers/recurrent/test_gru.h"

int main()
{
    test_change_policy();
    test_policy_operations();
    test_policy_selector();
    
    test_named_params();
    
    test_eval_plan();
    
    test_cpu_matrix();
    test_one_hot_vector();
    test_trival_matrix();
    test_zero_matrix();
    
    test_abs();
    test_add();
    test_collapse();
    test_divide();
    test_dot();
    test_element_mul();
    test_interpolate();
    test_negative_log_likelihood();
    test_negative_log_likelihood_derivative();
    test_sigmoid();
    test_sigmoid_derivative();
    test_sign();
    test_softmax();
    test_softmax_derivative();
    test_substract();
    test_tanh();
    test_tanh_derivative();
    test_transpose();
    
    test_abs_layer();
    test_add_layer();
    test_bias_layer();
    test_element_mul_layer();
    test_interpolate_layer();
    test_sigmoid_layer();
    test_softmax_layer();
    test_tanh_layer();
    test_weight_layer();
    
    test_negative_log_likelihood_layer();
    
    test_compose_kernel();
    test_linear_layer();
    test_single_layer();
    
    test_gru();
}
