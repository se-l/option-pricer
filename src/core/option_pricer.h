#pragma once
#include <vector>
#include <string>
#include "../common/types.h"


API std::vector<float> get_v_iv_fd_cuda(
    const std::vector<float> &prices,
    const std::vector<float> &spots,
    const std::vector<float> &strikes,
    const std::vector<float> &tenors,
    const std::vector<std::string> &rights,
    float r,
    int n_steps = 200,
    const std::vector<float> &div_amounts = {},
    const std::vector<float> &div_times = {},
    float tol = 1e-6f,
    int max_iter = 100,
    float v_min = 1e-4f,
    float v_max = 5.0f
    );

API float get_single_iv_cpu(
    float price,
    float spot,
    float strike,
    float tenor,
    bool is_call,
    float r,
    const std::vector<float> &div_amounts = {},
    const std::vector<float> &div_times = {},
    float tol = 1e-6f,
    int max_iter = 100,
    float steps_factor = 1
    );

API float get_single_iv_cuda(
    float price,
    float spot,
    float strike,
    float tenor,
    bool is_call,
    float r,
    int n_steps = 200,
    const std::vector<float> &div_amounts = {},
    const std::vector<float> &div_times = {},
    float tol = 1e-6f,
    int max_iter = 100
    );

API std::vector<float> get_v_iv_fd_with_term_structure_cuda(
    const std::vector<float> &prices,
    const std::vector<float> &spots,
    const std::vector<float> &strikes,
    const std::vector<float> &tenors,
    const std::vector<std::string> &rights,
    const std::vector<float> &r_curve,
    const std::vector<float> &time_points,
    int n_steps = 200,
    const std::vector<float> &div_amounts = {},
    const std::vector<float> &div_times = {}
);

API std::vector<float> get_v_fd_price_cuda(
    const std::vector<float> &spots,
    const std::vector<float> &strikes,
    const std::vector<float> &tenors,
    const std::vector<std::string> &rights,
    const std::vector<float> &sigmas,
    float r,
    int n_steps = 100,
    const std::vector<float> &div_amounts = {},
    const std::vector<float> &div_times = {});

API std::vector<float> get_v_fd_delta_cuda(
    const std::vector<float> &spots,
    const std::vector<float> &strikes,
    const std::vector<float> &tenors,
    const std::vector<std::string> &rights,
    const std::vector<float> &sigmas,
    float r,
    int n_steps = 100,
    const std::vector<float> &div_amounts = {},
    const std::vector<float> &div_times = {});

API std::vector<float> get_v_fd_vega_cuda(
    const std::vector<float> &spots,
    const std::vector<float> &strikes,
    const std::vector<float> &tenors,
    const std::vector<std::string> &rights,
    const std::vector<float> &sigmas,
    float r,
    int n_steps = 100,
    const std::vector<float> &div_amounts = {},
    const std::vector<float> &div_times = {});
