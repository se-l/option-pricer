#pragma once

#ifdef _WIN32
#define MERLIN_EXPORT __declspec(dllexport)
#else
#define MERLIN_EXPORT
#endif

extern "C" {
    // Shared interface for all vectorized functions
    // v_is_call: 1 for Call, 0 for Put

    MERLIN_EXPORT float merlin_get_single_iv_cpu(
        float price, float spot, float strike, float tenor, uint8_t is_call,
        const float* rates_curve, const float* rates_times, int rates_count,
        const float* div_amounts, const float* div_times, int div_count,
        float tol = 1e-6f, int max_iter = 100, float steps_factor = 1
    );

    MERLIN_EXPORT void merlin_get_iv_fd_cuda(
        float* out_ivs,
        const float* prices, const float* spots, const float* strikes, const float* tenors, const uint8_t* v_is_call,
        int count,
        const float* rates_curve, const float* rates_times, int rates_count,
        const float* div_amounts, const float* div_times, int div_count,
        float tol = 1e-6f, int max_iter = 100, float v_min = 1e-4f, float v_max = 5.0f, float steps_factor = 1.0f
    );

    MERLIN_EXPORT void merlin_get_price_fd_cuda(
        float* out_prices,
        const float* spots, const float* strikes, const float* tenors, const uint8_t* v_is_call, const float* sigmas,
        int count, int n_steps,
        const float* rates_curve, const float* rates_times, int rates_count,
        const float* div_amounts, const float* div_times, int div_count
    );

    MERLIN_EXPORT void merlin_get_delta_fd_cuda(
        float* out_deltas,
        const float* spots, const float* strikes, const float* tenors, const uint8_t* v_is_call, const float* sigmas,
        int count, int n_steps,
        const float* rates_curve, const float* rates_times, int rates_count,
        const float* div_amounts, const float* div_times, int div_count
    );

    MERLIN_EXPORT void merlin_get_vega_fd_cuda(
        float* out_vegas,
        const float* spots, const float* strikes, const float* tenors, const uint8_t* v_is_call, const float* sigmas,
        int count, int n_steps,
        const float* rates_curve, const float* rates_times, int rates_count,
        const float* div_amounts, const float* div_times, int div_count
    );
}