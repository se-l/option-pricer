#pragma once
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
    const std::vector<float> &div_times = {});

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
