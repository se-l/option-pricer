#pragma once
#include "../common/types.h"


API std::vector<float> f_min_price_surface_theta_rho_psi_cuda(
    const std::vector<float>& calibration_params,
    const std::vector<float>& item_prices,
    const std::vector<float>& s,
    const std::vector<float>& k,
    const std::vector<float>& t,
    const std::vector<std::string>& rights,
    const std::vector<float>& mny_fwd_ln,
    const std::vector<int>& tenor_index,
    float r,
    int n_steps,
    const std::vector<float>& div_amounts,
    const std::vector<float>& div_times
);