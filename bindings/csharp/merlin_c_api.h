#pragma once

#include "../../src/common/types.h"

#define MERLIN_C_API API

extern "C" {
    // Shared interface for all vectorized functions
    // v_is_call: 1 for Call, 0 for Put

    MERLIN_C_API float merlin_implied_vol_american_fd_host(
        float price, float spot, float strike, float tenor, uint8_t is_call,
        const float* rates_curve, const float* rates_times, int rates_count,
        const float* div_amounts, const float* div_times, int div_count,
        float tol = 1e-6f,
        int max_iter = 100,
        float v_min = 1e-4f,
        float v_max = 5.0f,
        int time_steps = 200,
        int space_steps = 200
    );

    MERLIN_C_API void merlin_get_iv_fd_cpu(
        float* out_ivs,
        const float* prices, const float* spots, const float* strikes, const float* tenors, const uint8_t* v_is_call,
        int count,
        const float* rates_curve, const float* rates_times, int rates_count,
        const float* div_amounts, const float* div_times, int div_count,
        float tol = 1e-6f, int max_iter = 100, float v_min = 1e-4f, float v_max = 5.0f,
        int time_steps = 200, int space_steps = 200
    );

    MERLIN_C_API void merlin_get_iv_fd_gpu(
        float* out_ivs,
        const float* prices, const float* spots, const float* strikes, const float* tenors, const uint8_t* v_is_call,
        int count,
        const float* rates_curve, const float* rates_times, int rates_count,
        const float* div_amounts, const float* div_times, int div_count,
        float tol = 1e-6f, int max_iter = 100, float v_min = 1e-4f, float v_max = 5.0f,
        int time_steps = 200, int space_steps = 200
    );

    MERLIN_C_API float merlin_price_american_fd_div_cpu(
        float spot,
        float strike,
        float tenor,
        float sigma,
        uint8_t is_call,
        const float* rates_curve, const float* rates_times, int rates_count,
        const float* div_amounts, const float* div_times, int div_count,
        int time_steps = 200,
        int space_steps = 200
    );

    MERLIN_C_API void merlin_get_price_fd_cuda(
        float* out_prices,
        const float* spots, const float* strikes, const float* tenors, const uint8_t* v_is_call, const float* sigmas,
        int count, int n_steps,
        const float* rates_curve, const float* rates_times, int rates_count,
        const float* div_amounts, const float* div_times, int div_count
    );

    MERLIN_C_API void merlin_get_iv_binomial_cuda(
        float* out_ivs,
        const float* prices, const float* spots, const float* strikes, const float* tenors, const uint8_t* v_is_call,
        int count,
        const float* rates_curve, const float* rates_times, int rates_count,
        const float* div_amounts, const float* div_times, int div_count,
        float tol = 1e-6f, int max_iter = 100, float v_min = 1e-4f, float v_max = 5.0f, float steps_factor = 1.0f
    );

    MERLIN_C_API void merlin_get_delta_fd_cuda(
        float* out_deltas,
        const float* spots, const float* strikes, const float* tenors, const uint8_t* v_is_call, const float* sigmas,
        int count, int n_steps,
        const float* rates_curve, const float* rates_times, int rates_count,
        const float* div_amounts, const float* div_times, int div_count
    );

    MERLIN_C_API void merlin_get_vega_fd_cuda(
        float* out_vegas,
        const float* spots, const float* strikes, const float* tenors, const uint8_t* v_is_call, const float* sigmas,
        int count, int n_steps,
        const float* rates_curve, const float* rates_times, int rates_count,
        const float* div_amounts, const float* div_times, int div_count
    );
}