#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
#include "../pricing_engine.h"

// Helper: E-SSVI total variance
static inline float essvi_total_variance(float mny_fwd_ln, float theta, float rho, float psi) {
    // Clamp inputs to avoid numerical issues
    const float eps_theta = 1e-10f;
    theta = std::max(theta, eps_theta);
    rho = std::min(0.999f, std::max(-0.999f, rho));
    psi = std::max(0.0f, psi);

    // phi = psi / sqrt(theta)
    float phi = psi / sqrtf(theta);

    // SSVI total variance formula (Gatheral-Jacquier)
    // w(k) = theta/2 * [ 1 + rho * phi * k + sqrt( (phi * k + rho)^2 + 1 - rho^2 ) ]
    float x = phi * mny_fwd_ln + rho;
    float rad = sqrtf(x * x + (1.0f - rho * rho));
    float w = 0.5f * theta * (1.0f + rho * phi * mny_fwd_ln + rad);
    return std::max(w, 0.0f);
}

// Build E-SSVI IVs per option from tenor-params, then price and return residuals
std::vector<float> f_min_price_surface_theta_rho_psi_cuda(
    const std::vector<float>& calibration_params,
    const std::vector<float>& item_prices,
    const std::vector<float>& s,
    const std::vector<float>& k,
    const std::vector<float>& t,
    const std::vector<uint8_t>& v_is_call,
    const std::vector<float>& mny_fwd_ln,
    const std::vector<int>& tenor_index,
    const int n_steps = 200,
    const std::vector<float>& rates_curve = {},
    const std::vector<float>& rates_times = {},
    const std::vector<float>& div_amounts = {},
    const std::vector<float>& div_times = {}
) {
    const size_t n_opts = s.size();
    if (k.size() != n_opts || t.size() != n_opts || v_is_call.size() != n_opts ||
        mny_fwd_ln.size() != n_opts || item_prices.size() != n_opts || tenor_index.size() != n_opts) {
        throw std::runtime_error("Input size mismatch in f_min_price_surface_theta_rho_psi_cuda");
    }
    if (calibration_params.size() % 3 != 0) {
        throw std::runtime_error("calibration_params must have size multiple of 3 (theta, rho, psi per tenor)");
    }
    const int n_tenors = static_cast<int>(calibration_params.size() / 3);

    // Build per-option IV from E-SSVI total variance
    std::vector<float> model_iv;
    model_iv.reserve(n_opts);
    for (size_t i = 0; i < n_opts; ++i) {
        int ti = tenor_index[i];
        if (ti < 0 || ti >= n_tenors) {
            throw std::runtime_error("tenor_index out of range");
        }
        float theta = calibration_params[3 * ti + 0];
        float rho   = calibration_params[3 * ti + 1];
        float psi   = calibration_params[3 * ti + 2];

        float w = essvi_total_variance(mny_fwd_ln[i], theta, rho, psi);
        float Ti = std::max(t[i], 1e-8f);
        float iv = sqrtf(std::max(w, 0.0f) / Ti);
        // Guard against NaN/Inf
        if (!std::isfinite(iv)) iv = 0.0f;
        model_iv.push_back(iv);
    }

    // Price using existing CUDA pricer
    std::vector<float> model_prices = get_v_fd_price_cuda(
        s, k, t, model_iv, v_is_call,
        rates_curve, rates_times,
        div_amounts, div_times
    );

    // Residuals = model - market
    std::vector<float> residuals(n_opts, 0.0f);
    for (size_t i = 0; i < n_opts; ++i) {
        float res = model_prices[i] - item_prices[i];
        if (!std::isfinite(res)) {
            // Replace NaN/Inf residuals with 0 (or consider using a large penalty if preferred)
            res = 0.0f;
        }
        residuals[i] = res;
    }
    return residuals;
}
