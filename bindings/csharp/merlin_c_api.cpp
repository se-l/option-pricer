#include <vector>
#include <string>
#include "./merlin_c_api.h"
#include "../../src/core/pricing_engine.h"

// Helper to convert C-style pointers to std::vectors
template<typename T>
std::vector<T> to_vec(const T* ptr, int count) {
    return (ptr && count > 0) ? std::vector<T>(ptr, ptr + count) : std::vector<T>();
}

extern "C" {

float merlin_implied_vol_american_fd_host(
    const float price,
    const float spot,
    const float strike,
    const float tenor,
    const uint8_t is_call,
    const float* rates_curve,
    const float* rates_times,
    const int n_rates,
    const float* div_amounts,
    const float* div_times,
    const int n_divs,
    const float tol,
    const int max_iter,
    const float v_min,
    const float v_max,
    const int time_steps,
    const int space_steps
) {
    return implied_vol_american_fd_host(
        price, spot, strike, tenor, is_call,
        to_vec(rates_curve, n_rates),
        to_vec(rates_times, n_rates),
        to_vec(div_amounts, n_divs),
        to_vec(div_times, n_divs),
        tol,
        max_iter,
        v_min,
        v_max,
        time_steps,
        space_steps
        );
}

auto merlin_get_iv_fd_cpu(
    float *out_ivs,
    const float *prices,
    const float *spots,
    const float *strikes,
    const float *tenors,
    const uint8_t *v_is_call,
    const int count,
    const float *rates_curve,
    const float *rates_times,
    const int n_rates,
    const float *div_amounts,
    const float *div_times,
    const int n_divs,
    const float tol,
    const int max_iter,
    const float v_min,
    const float v_max,
    const int time_steps,
    const int space_steps
) -> void {
    // Defensive check: If pointers are null but count > 0, we would crash/read garbage
    if (count > 0 && (!prices || !spots || !strikes || !tenors || !v_is_call || !out_ivs)) {
        return;
    }

    std::vector<float> results = get_v_iv_fd_cpu(
        to_vec(prices, count),
        to_vec(spots, count),
        to_vec(strikes, count),
        to_vec(tenors, count),
        to_vec(v_is_call, count),
        to_vec(rates_curve, n_rates),
        to_vec(rates_times, n_rates),
        to_vec(div_amounts, n_divs),
        to_vec(div_times, n_divs),
        tol, max_iter, v_min, v_max, time_steps, space_steps
    );

    // Check if output buffer is safe to write to
    if (!results.empty()) {
        std::copy(results.begin(), results.end(), out_ivs);
    }
}

void merlin_get_iv_fd_gpu(
    float* out_ivs,
    const float* prices,
    const float* spots,
    const float* strikes,
    const float* tenors,
    const uint8_t* is_calls,
    const int count,
    const float* rates_curve,
    const float* rates_times,
    const int n_rates,
    const float* div_amounts,
    const float* div_times,
    const int n_divs,
    const float tol,
    const int max_iter,
    const float v_min,
    const float v_max,
    const int time_steps,
    const int space_steps
) {
    std::vector<float> results = get_v_iv_fd_cuda_new(
        to_vec(prices, count),
        to_vec(spots, count),
        to_vec(strikes, count),
        to_vec(tenors, count),
        to_vec(is_calls, count),
        to_vec(rates_curve, n_rates),
        to_vec(rates_times, n_rates),
        to_vec(div_amounts, n_divs),
        to_vec(div_times, n_divs),
        tol, max_iter, v_min, v_max, time_steps, space_steps
    );

    std::copy(results.begin(), results.end(), out_ivs);
}

float merlin_price_american_fd_div_cpu(
    float spot,
    float strike,
    float tenor,
    float sigma,
    uint8_t is_call,
    const float* rates_curve,
    const float* rates_times,
    int n_rates,
    const float* div_amounts,
    const float* div_times,
    int n_divs,
    int time_steps,
    int space_steps
) {
    return price_american_fd_div_host(
        spot, strike, tenor, sigma, is_call,
        to_vec(rates_curve, n_rates),
        to_vec(rates_times, n_rates),
        to_vec(div_amounts, n_divs),
        to_vec(div_times, n_divs),
        time_steps, space_steps
    );
}

} // extern "C"