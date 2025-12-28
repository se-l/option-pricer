#include <vector>
#include <string>
#include "./merlin_c_api.h"
#include "../../src/core/option_pricer.h"

// Helper to convert C-style pointers to std::vectors
template<typename T>
std::vector<T> to_vec(const T* ptr, int count) {
    return (ptr && count > 0) ? std::vector<T>(ptr, ptr + count) : std::vector<T>();
}

// Helper for rights conversion (int -> string)
std::vector<std::string> to_rights_vec(const int* rights, int count) {
    std::vector<std::string> v(count);
    for (int i = 0; i < count; ++i) {
        v[i] = (rights[i] == 1) ? "c" : "p";
    }
    return v;
}

extern "C" {

float merlin_get_single_iv_cpu(float price, float spot, float strike,
                      float tenor, bool is_call, float r,
                      const float* div_amounts, const float* div_times, int div_count,
                      float tol, int max_iter, float steps_factor)
{
    std::vector<float> d_amounts = to_vec(div_amounts, div_count);
    std::vector<float> d_times = to_vec(div_times, div_count);

    return get_single_iv_cpu(
        price, spot, strike, tenor, is_call, r,
        d_amounts, d_times, tol, max_iter, steps_factor
    );
}

float merlin_get_single_iv_cuda(const float price, const float spot, const float strike,
                      const float tenor, const bool is_call, float r, int n_steps,
                      const float* div_amounts, const float* div_times, int div_count,
                      float tol, int max_iter)
{
    // Convert common dividends to vector once
    std::vector<float> d_amounts;
    std::vector<float> d_times;
    if (div_count > 0) {
        d_amounts.assign(div_amounts, div_amounts + div_count);
        d_times.assign(div_times, div_times + div_count);
    }
    return get_single_iv_cuda(
        price,
        spot,
        strike,
        tenor,
        is_call,
        r,
        n_steps,
        d_amounts,
        d_times,
        tol,
        max_iter // This maps to max_outer_iters in the new signature
    );
}

void merlin_get_iv_fd(float* out_ivs, const float* prices, const float* spots, const float* strikes,
                      const float* tenors, const int* rights, int count, float r, int n_steps,
                      const float* div_amounts, const float* div_times, int div_count,
                      float tol, int max_iter, float v_min, float v_max) {
    auto res = get_v_iv_fd_cuda(to_vec(prices, count), to_vec(spots, count), to_vec(strikes, count),
                               to_vec(tenors, count), to_rights_vec(rights, count),
                               r, n_steps, to_vec(div_amounts, div_count), to_vec(div_times, div_count),
                               tol, max_iter, v_min, v_max);
    std::copy(res.begin(), res.end(), out_ivs);
}

void merlin_get_iv_fd_term_structure(float* out_ivs, const float* prices, const float* spots, const float* strikes,
                                     const float* tenors, const int* rights, int count,
                                     const float* r_curve, const float* time_points, int n_curve_points, int n_steps,
                                     const float* div_amounts, const float* div_times, int div_count) {
    auto res = get_v_iv_fd_with_term_structure_cuda(to_vec(prices, count), to_vec(spots, count), to_vec(strikes, count),
                                                   to_vec(tenors, count), to_rights_vec(rights, count),
                                                   to_vec(r_curve, n_curve_points), to_vec(time_points, n_curve_points),
                                                   n_steps, to_vec(div_amounts, div_count), to_vec(div_times, div_count));
    std::copy(res.begin(), res.end(), out_ivs);
}

void merlin_get_price_fd(float* out_prices, const float* spots, const float* strikes, const float* tenors,
                         const int* rights, const float* sigmas, int count, float r, int n_steps,
                         const float* div_amounts, const float* div_times, int div_count) {
    auto res = get_v_fd_price_cuda(to_vec(spots, count), to_vec(strikes, count), to_vec(tenors, count),
                                  to_rights_vec(rights, count), to_vec(sigmas, count),
                                  r, n_steps, to_vec(div_amounts, div_count), to_vec(div_times, div_count));
    std::copy(res.begin(), res.end(), out_prices);
}

void merlin_get_delta_fd(float* out_deltas, const float* spots, const float* strikes, const float* tenors,
                         const int* rights, const float* sigmas, int count, float r, int n_steps,
                         const float* div_amounts, const float* div_times, int div_count) {
    auto res = get_v_fd_delta_cuda(to_vec(spots, count), to_vec(strikes, count), to_vec(tenors, count),
                                  to_rights_vec(rights, count), to_vec(sigmas, count),
                                  r, n_steps, to_vec(div_amounts, div_count), to_vec(div_times, div_count));
    std::copy(res.begin(), res.end(), out_deltas);
}

void merlin_get_vega_fd(float* out_vegas, const float* spots, const float* strikes, const float* tenors,
                        const int* rights, const float* sigmas, int count, float r, int n_steps,
                        const float* div_amounts, const float* div_times, int div_count) {
    auto res = get_v_fd_vega_cuda(to_vec(spots, count), to_vec(strikes, count), to_vec(tenors, count),
                                 to_rights_vec(rights, count), to_vec(sigmas, count),
                                 r, n_steps, to_vec(div_amounts, div_count), to_vec(div_times, div_count));
    std::copy(res.begin(), res.end(), out_vegas);
}

} // extern "C"