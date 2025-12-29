#include <vector>
#include <string>
#include "./merlin_c_api.h"
#include "../../src/core/option_pricer.h"

// Helper to convert C-style pointers to std::vectors
template<typename T>
std::vector<T> to_vec(const T* ptr, int count) {
    return (ptr && count > 0) ? std::vector<T>(ptr, ptr + count) : std::vector<T>();
}

extern "C" {

API float merlin_get_single_iv_cpu(
    float price,
    float spot,
    float strike,
    float tenor,
    uint8_t is_call,
    const float* rates_curve,
    const float* rates_times,
    int n_rates,
    const float* div_amounts,
    const float* div_times,
    int n_divs,
    float tol,
    int max_iter,
    float steps_factor
) {
    return get_single_iv_cpu(
        price, spot, strike, tenor, is_call,
        to_vec(rates_curve, n_rates),
        to_vec(rates_times, n_rates),
        to_vec(div_amounts, n_divs),
        to_vec(div_times, n_divs),
        tol, max_iter, steps_factor
    );
}

API void merlin_get_iv_fd_cuda(
    float* out_ivs,
    const float* prices,
    const float* spots,
    const float* strikes,
    const float* tenors,
    const uint8_t* is_calls,
    int count,
    const float* rates_curve,
    const float* rates_times,
    int n_rates,
    const float* div_amounts,
    const float* div_times,
    int n_divs,
    float tol,
    int max_iter,
    float v_min,
    float v_max,
    float steps_factor
) {
    std::vector<float> results = get_v_iv_fd_cuda(
        to_vec(prices, count),
        to_vec(spots, count),
        to_vec(strikes, count),
        to_vec(tenors, count),
        to_vec(is_calls, count),
        to_vec(rates_curve, n_rates),
        to_vec(rates_times, n_rates),
        to_vec(div_amounts, n_divs),
        to_vec(div_times, n_divs),
        tol, max_iter, v_min, v_max, steps_factor
    );

    std::copy(results.begin(), results.end(), out_ivs);
}

} // extern "C"