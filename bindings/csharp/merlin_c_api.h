#pragma once

#ifdef _WIN32
#define MERLIN_EXPORT __declspec(dllexport)
#else
#define MERLIN_EXPORT
#endif

extern "C" {
    // Shared interface for all vectorized functions
    // rights: 1 for Call, 0 for Put

    MERLIN_EXPORT float merlin_get_single_iv_cpu(
        float price, float spot, float strike, float tenor, bool is_call, float r,
        const float* div_amounts, const float* div_times, int div_count,
        float tol = 1e-6f, int max_iter = 100, float steps_factor = 1
    );

    MERLIN_EXPORT float merlin_get_single_iv_cuda(
        float price, float spot, float strike, float tenor, bool is_call,
        float r, int n_steps,
        const float* div_amounts, const float* div_times, int div_count,
        float tol = 1e-6f, int max_iter = 100
    );

    MERLIN_EXPORT void merlin_get_iv_fd(
        float* out_ivs,
        const float* prices, const float* spots, const float* strikes, const float* tenors, const int* rights,
        int count, float r, int n_steps,
        const float* div_amounts, const float* div_times, int div_count,
        float tol = 1e-6f, int max_iter = 100, float v_min = 1e-4f, float v_max = 5.0f
    );

    MERLIN_EXPORT void merlin_get_iv_fd_term_structure(
        float* out_ivs,
        const float* prices, const float* spots, const float* strikes, const float* tenors, const int* rights,
        int count, const float* r_curve, const float* time_points, int n_curve_points, int n_steps,
        const float* div_amounts, const float* div_times, int div_count
    );

    MERLIN_EXPORT void merlin_get_price_fd(
        float* out_prices,
        const float* spots, const float* strikes, const float* tenors, const int* rights, const float* sigmas,
        int count, float r, int n_steps,
        const float* div_amounts, const float* div_times, int div_count
    );

    MERLIN_EXPORT void merlin_get_delta_fd(
        float* out_deltas,
        const float* spots, const float* strikes, const float* tenors, const int* rights, const float* sigmas,
        int count, float r, int n_steps,
        const float* div_amounts, const float* div_times, int div_count
    );

    MERLIN_EXPORT void merlin_get_vega_fd(
        float* out_vegas,
        const float* spots, const float* strikes, const float* tenors, const int* rights, const float* sigmas,
        int count, float r, int n_steps,
        const float* div_amounts, const float* div_times, int div_count
    );
}