#pragma once
#include <vector>
#include "../common/types.h"


API std::vector<float> get_v_iv_fd_cuda(
    const std::vector<float> &prices,
    const std::vector<float> &spots,
    const std::vector<float> &strikes,
    const std::vector<float> &tenors,
    const std::vector<uint8_t> &v_is_call,
    const std::vector<float> &rates_curve = {},
    const std::vector<float> &rates_times = {},
    const std::vector<float> &div_amounts = {},
    const std::vector<float> &div_times = {},
    float tol = 1e-6f,
    int max_iter = 200,
    float v_min = 1e-7f,
    float v_max = 5.0f,
    float steps_factor = 1
);

API float get_single_iv_cpu(
    float price,
    float spot,
    float strike,
    float tenor,
    uint8_t is_call,
    const std::vector<float> &rates_curve = {},
    const std::vector<float> &rates_times = {},
    const std::vector<float> &div_amounts = {},
    const std::vector<float> &div_times = {},
    float tol = 1e-6f,
    int max_iter = 100,
    float steps_factor = 1
    );

API std::vector<float> get_v_fd_price_cuda(
    const std::vector<float> &spots,
    const std::vector<float> &strikes,
    const std::vector<float> &tenors,
    const std::vector<uint8_t> &v_is_call,
    const std::vector<float> &sigmas,
    int n_steps = 100,
    const std::vector<float> &rates_curve = {},
    const std::vector<float> &rates_times = {},
    const std::vector<float> &div_amounts = {},
    const std::vector<float> &div_times = {});

API std::vector<float> get_v_fd_delta_cuda(
    const std::vector<float> &spots,
    const std::vector<float> &strikes,
    const std::vector<float> &tenors,
    const std::vector<uint8_t> &v_is_call,
    const std::vector<float> &sigmas,
    int n_steps = 100,
    const std::vector<float> &rates_curve = {},
    const std::vector<float> &rates_times = {},
    const std::vector<float> &div_amounts = {},
    const std::vector<float> &div_times = {});

API std::vector<float> get_v_fd_vega_cuda(
    const std::vector<float> &spots,
    const std::vector<float> &strikes,
    const std::vector<float> &tenors,
    const std::vector<uint8_t> &v_is_call,
    const std::vector<float> &sigmas,
    int n_steps = 100,
    const std::vector<float> &rates_curve = {},
    const std::vector<float> &rates_times = {},
    const std::vector<float> &div_amounts = {},
    const std::vector<float> &div_times = {});