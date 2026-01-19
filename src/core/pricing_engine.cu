#include "pricing_engine.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdexcept>


// Device version of tridiagonal solver (Thomas algorithm)
__device__ inline void solve_tridiagonal_cuda(
    int n,
    const float* a,  // sub-diagonal [1..n-1]
    const float* b,  // diagonal    [0..n-1]
    const float* c,  // super-diag  [0..n-2]
    float* d)        // RHS         [0..n-1], overwritten by solution
{
    // Use stack arrays for temporaries
    constexpr int MAX_GRID = 512;
    float c_star[MAX_GRID];
    float d_star[MAX_GRID];

    if (n > MAX_GRID) return; // Safety check

    c_star[0] = c[0] / b[0];
    d_star[0] = d[0] / b[0];

    for (int i = 1; i < n; ++i) {
        float m = 1.0f / (b[i] - a[i] * c_star[i - 1]);
        c_star[i] = (i < n - 1) ? (c[i] * m) : 0.0f;
        d_star[i] = (d[i] - a[i] * d_star[i - 1]) * m;
    }

    d[n - 1] = d_star[n - 1];
    for (int i = n - 2; i >= 0; --i) {
        d[i] = d_star[i] - c_star[i] * d[i + 1];
    }
}


__device__ inline float get_rate_at_time(float t, const float *rates_curve, const float *rates_times, int n_points) {
    if (n_points <= 0) return 0.0f;
    if (t <= rates_times[0]) return rates_curve[0];
    if (t >= rates_times[n_points - 1]) {
        // Beyond the last point, we usually assume the last instantaneous forward rate continues
        // For simplicity, return the last zero rate, but ideally calculate the last forward.
        return rates_curve[n_points - 1];
    }

    // Find the interval [i, i+1]
    for (int i = 0; i < n_points - 1; ++i) {
        if (t <= rates_times[i + 1]) {
            // Calculate the Instantaneous Forward Rate for this segment
            // f = [R(ti+1)*ti+1 - R(ti)*ti] / (ti+1 - ti)
            float t_low = rates_times[i];
            float t_high = rates_times[i + 1];
            float r_low = rates_curve[i];
            float r_high = rates_curve[i + 1];

            // This is the constant forward rate valid for the entire duration of this segment
            return (r_high * t_high - r_low * t_low) / (t_high - t_low);
        }
    }
    return rates_curve[n_points - 1];
}

__device__ float calculate_pv_of_dividends(
    float T,
    const float* rates_curve,
    const float* rates_times,
    int n_rates,
    const float* div_amounts,
    const float* div_times,
    int n_divs
    ) {
    float pv_divs = 0.0f;
    for (int i = 0; i < n_divs; ++i) {
        float t_div = div_times[i];
        if (t_div > 0.0f && t_div <= T) {
            float r_div = get_rate_at_time(t_div, rates_curve, rates_times, n_rates);
            pv_divs += div_amounts[i] * expf(-r_div * t_div);
        }
    }
}

// Build dividend schedule
struct DivEvent {
    int step;
    float amount;
    float time;
};
constexpr int MAX_DIV_EVENTS = 32;

__device__ void precompute_dividends_cuda(
        float T, int time_steps,
        const float* rates_curve, const float* rates_times, int n_rates,
        const float* div_amounts, const float* div_times, int n_divs,
        float& out_pv_divs, DivEvent* out_div_events, int& out_n_div_events) {

    out_pv_divs = 0.0f;
    for (int i = 0; i < n_divs; ++i) {
        float t_div = div_times[i];
        if (t_div > 0.0f && t_div <= T) {
            float r_div = get_rate_at_time(t_div, rates_curve, rates_times, n_rates);
            out_pv_divs += div_amounts[i] * expf(-r_div * t_div);
        }
    }

    out_n_div_events = 0;
    float dt = T / static_cast<float>(time_steps);

    for (int k = 0; k < n_divs && out_n_div_events < MAX_DIV_EVENTS; ++k) {
        float tD = div_times[k];
        if (tD > 0.0f && tD < T) {
            int step = static_cast<int>(floorf(tD / dt));
            if (step < 0) step = 0;
            if (step >= time_steps) step = time_steps - 1;
            out_div_events[out_n_div_events++] = {step, div_amounts[k], tD};
        }
    }
}

__device__ float price_american_fd_div_cuda(
    float S, float K, float T,
    float sigma, uint8_t is_call,
    const float* rates_curve,
    const float* rates_times,
    int n_rates,
    float pv_divs,
    const DivEvent* div_events,
    int n_div_events,
    int time_steps = 200,
    int space_steps = 200
    )
{
    constexpr int MAX_TIME_STEPS = 512;
    constexpr int MAX_SPACE_STEPS = 512;

    if (T <= 0.0f || S <= 0.0f || K <= 0.0f || sigma <= 0.0f)
        return is_call ? fmaxf(S - K, 0.0f) : fmaxf(K - S, 0.0f);

    // Bound the grid sizes
    const int N_t = (time_steps < 10) ? 10 : ((time_steps > MAX_TIME_STEPS) ? MAX_TIME_STEPS : time_steps);
    const int N_s = (space_steps < 10) ? 10 : ((space_steps > MAX_SPACE_STEPS) ? MAX_SPACE_STEPS : space_steps);

    const float dt = T / static_cast<float>(N_t);

    // float S_grid_center = fmaxf(S - pv_divs, S * 0.1f);

    // Build spatial grid - match host version exactly
    float r0 = get_rate_at_time(0.0f, rates_curve, rates_times, n_rates);

    // For puts, ensure grid goes low enough to capture deep ITM scenarios
    float S_min = 0.0f;  // Always start at zero for puts
    float S_max = fmaxf(
        (S + pv_divs) * expf((r0 + 4.0f * sigma) * T),
        3.0f * fmaxf(S, K)
    );

    const float dS = (S_max - S_min) / static_cast<float>(N_s);

    // Stock price grid (stack array)
    float S_grid[MAX_SPACE_STEPS + 1];
    for (int i = 0; i <= N_s; ++i) {
        S_grid[i] = S_min + i * dS;
    }

    // Terminal condition and working arrays
    float V[MAX_SPACE_STEPS + 1];
    float a[MAX_SPACE_STEPS + 1], b[MAX_SPACE_STEPS + 1], c[MAX_SPACE_STEPS + 1];
    float rhs[MAX_SPACE_STEPS + 1];

    for (int i = 0; i <= N_s; ++i) {
        float payoff = is_call
            ? fmaxf(S_grid[i] - K, 0.0f)
            : fmaxf(K - S_grid[i], 0.0f);
        V[i] = payoff;
    }

    // Backward time stepping
    for (int n = N_t - 1; n >= 0; --n) {
        float t = n * dt;
        float r = get_rate_at_time(t, rates_curve, rates_times, n_rates);

        // Crank-Nicolson with theta = 0.5
        const float theta = 0.5f;
        const float sigma2 = sigma * sigma;

        // Interior points: i = 1 to N_s - 1
        for (int i = 1; i < N_s; ++i) {
            float Si = S_grid[i];
            if (Si < 1e-8f) continue; // Skip near-zero nodes

            float Si2 = Si * Si;

            // Standard FD coefficients in price space
            float coef_pp = 0.5f * sigma2 * Si2 / (dS * dS);  // S²σ²/2
            float coef_p = 0.5f * r * Si / dS;                 // rS/2
            float coef_0 = -sigma2 * Si2 / (dS * dS) - r;     // -S²σ² - r

            float alpha = coef_pp - coef_p;   // coefficient of V[i-1]
            float beta = coef_0;               // coefficient of V[i]
            float gamma = coef_pp + coef_p;    // coefficient of V[i+1]

            // LHS (implicit)
            a[i] = -theta * dt * alpha;
            b[i] = 1.0f - theta * dt * beta;
            c[i] = -theta * dt * gamma;

            // RHS (explicit)
            rhs[i] = V[i] * (1.0f + (1.0f - theta) * dt * beta)
                   + V[i-1] * (1.0f - theta) * dt * alpha
                   + V[i+1] * (1.0f - theta) * dt * gamma;
        }

        // Lower boundary: S = 0
        {
            a[0] = 0.0f;
            b[0] = 1.0f;
            c[0] = 0.0f;

            float t_remaining = T - t;
            if (is_call) {
                rhs[0] = 0.0f;
            } else {
                // Put at S=0: worth K*exp(-r*(T-t))
                rhs[0] = K * expf(-r * t_remaining);
            }
        }

        // Upper boundary: S = S_max
        {
            a[N_s] = 0.0f;
            b[N_s] = 1.0f;
            c[N_s] = 0.0f;

            float t_remaining = T - t;
            if (is_call) {
                // Call at large S: V ≈ S - K*exp(-r*(T-t))
                rhs[N_s] = S_grid[N_s] - K * expf(-r * t_remaining);
            } else {
                // Put at large S: worth approximately 0
                rhs[N_s] = 0.0f;
            }
        }

        // Solve tridiagonal
        solve_tridiagonal_cuda(N_s + 1, a, b, c, rhs);
        for (int i = 0; i <= N_s; ++i) {
            V[i] = rhs[i];
        }

        // American early exercise at time t (before dividend)
        for (int i = 0; i <= N_s; ++i) {
            float intrinsic = is_call
                ? fmaxf(S_grid[i] - K, 0.0f)
                : fmaxf(K - S_grid[i], 0.0f);
            V[i] = fmaxf(V[i], intrinsic);
        }

        // Apply dividend jumps
        for (int div_idx = 0; div_idx < n_div_events; ++div_idx) {
            if (div_events[div_idx].step == n) {
                float V_new[MAX_SPACE_STEPS + 1];

                for (int i = 0; i <= N_s; ++i) {
                    float S_pre_div = S_grid[i];
                    float S_post_div = S_pre_div - div_events[div_idx].amount;

                    if (S_post_div <= 0.0f) {
                        // Stock worthless after dividend
                        V_new[i] = is_call ? 0.0f : K;
                    } else if (S_post_div >= S_max) {
                        V_new[i] = V[N_s];
                    } else if (S_post_div <= S_min) {
                        V_new[i] = V[0];
                    } else {
                        // Linear interpolation
                        float idx_f = (S_post_div - S_min) / dS;
                        int idx = static_cast<int>(floorf(idx_f));
                        if (idx < 0) idx = 0;
                        if (idx >= N_s) idx = N_s - 1;
                        float w = idx_f - idx;
                        V_new[i] = (1.0f - w) * V[idx] + w * V[idx + 1];
                    }
                }

                // Copy back
                for (int i = 0; i <= N_s; ++i) {
                    V[i] = V_new[i];
                }

                // Re-apply early exercise after dividend
                for (int i = 0; i <= N_s; ++i) {
                    float intrinsic = is_call
                        ? fmaxf(S_grid[i] - K, 0.0f)
                        : fmaxf(K - S_grid[i], 0.0f);
                    V[i] = fmaxf(V[i], intrinsic);
                }
            }
        }
    }

    // Final interpolation at spot S
    if (S <= S_min) return V[0];
    if (S >= S_max) return V[N_s];

    float idx_f = (S - S_min) / dS;
    int idx = static_cast<int>(floorf(idx_f));
    if (idx < 0) idx = 0;
    if (idx >= N_s) idx = N_s - 1;
    float w = idx_f - idx;

    return (1.0f - w) * V[idx] + w * V[idx + 1];
}

__device__ float implied_vol_american_fd_cuda(
    float target, float S, float K, float T, uint8_t is_call,
    const float* rates_curve,
    const float* rates_times,
    int n_rates,
    float pv_divs,
    const DivEvent* div_events,
    int n_div_events,
    float tol         = 1e-6f,
    int   max_iter    = 100,
    float v_min       = 1e-4f,
    float v_max       = 5.0f,
    int   time_steps  = 200,
    int   space_steps = 200)
{
    if (T <= 0.0f || S <= 0.0f || K <= 0.0f || target <= 0.0f)
        return 0.0f;

    // Check if price is at intrinsic
    float intrinsic = is_call ? fmaxf(S - K, 0.0f) : fmaxf(K - S, 0.0f);
    if (fabsf(target - intrinsic) < 1e-6f) {
        return 0.0f;
    }

    // Initial bracket check
    float p_low = price_american_fd_div_cuda(
        S, K, T, v_min, is_call,
        rates_curve, rates_times, n_rates,
        pv_divs, div_events, n_div_events,
        time_steps, space_steps);

    float p_high = price_american_fd_div_cuda(
        S, K, T, v_max, is_call,
        rates_curve, rates_times, n_rates,
        pv_divs, div_events, n_div_events,
        time_steps, space_steps);

    // Target below minimum possible price
    if (target <= p_low + 1e-9f)
        return v_min;

    // Target above maximum possible price
    if (target >= p_high - 1e-9f)
        return NAN;

    // Bisection with adaptive tolerance
    float lo = v_min;
    float hi = v_max;
    float mid = 0.0f;

    // Use relative tolerance for better convergence
    float rel_tol = tol / fmaxf(target, 1.0f);

    for (int i = 0; i < max_iter; ++i) {
        mid = 0.5f * (lo + hi);

        float p_mid = price_american_fd_div_cuda(
            S, K, T, mid, is_call,
            rates_curve, rates_times, n_rates,
            pv_divs, div_events, n_div_events,
            time_steps, space_steps);

        float diff = p_mid - target;
        float rel_diff = fabsf(diff) / target;

        // Check convergence with both absolute and relative tolerance
        if (fabsf(diff) < tol || rel_diff < rel_tol) {
            return mid;
        }

        if (diff < 0.0f) {
            lo = mid;
        } else {
            hi = mid;
        }

        // Check if bracket is tight enough
        if (hi - lo < v_min * 0.01f) {
            break;
        }
    }

    return mid;
}

// Kernel for FD pricing
__global__ void compute_fd_price_kernel(
    const float *spots, const float *strikes, const float *tenors, const float *sigmas,
    const uint8_t *v_is_call, int n_options,
    const float *rates_curve, const float *rates_times, int n_rates,
    const float *div_amounts, const float *div_times, int n_divs,
    float *out_price,
    int time_steps = 200, int space_steps = 200) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_options) return;

    float s = spots[idx];
    float k = strikes[idx];
    float t = tenors[idx];
    float sigma = sigmas[idx];
    uint8_t is_call = v_is_call[idx];

    if (!isfinite(s) || !isfinite(k) || !isfinite(t) || !isfinite(sigma) ||
        t <= 0.0f || s <= 0.0f || k <= 0.0f || sigma <= 0.0f) {
        out_price[idx] = NAN;
        return;
        }

    // Precalculate PV of dividends and Dividend Events (Once per thread/option)
    float pv_divs;
    DivEvent div_events[MAX_DIV_EVENTS];
    int n_div_events;

    precompute_dividends_cuda(t, time_steps, rates_curve, rates_times, n_rates,
                              div_amounts, div_times, n_divs,
                              pv_divs, div_events, n_div_events);

    out_price[idx] = price_american_fd_div_cuda(
        s, k, t, sigma, is_call,
        rates_curve, rates_times, n_rates,
        pv_divs, div_events, n_div_events,
        time_steps, space_steps);
}

// Kernel for FD IV calculation - refactored
__global__ void compute_fd_iv_kernel(
    const float *prices, const float *spots, const float *strikes, const float *tenors,
    const uint8_t *v_is_call, int n_options,
    const float *rates_curve, const float *rates_times, int n_rates,
    const float *div_amounts, const float *div_times, int n_divs,
    float *results,
    float tol, int max_iter, float v_min, float v_max,
    int time_steps, int space_steps) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_options) return;

    float p = prices[idx];
    float s = spots[idx];
    float k = strikes[idx];
    float t = tenors[idx];
    uint8_t is_call = v_is_call[idx];

    // Input validation
    if (!isfinite(p) || !isfinite(s) || !isfinite(k) || !isfinite(t)) {
        results[idx] = NAN;
        return;
    }

    if (t <= 0.0f || s <= 0.0f || k <= 0.0f || p <= 0.0f) {
        results[idx] = (t <= 0.0f) ? 0.0f : NAN;
        return;
    }

    // Check intrinsic value
    float intrinsic = is_call ? fmaxf(s - k, 0.0f) : fmaxf(k - s, 0.0f);
    if (fabsf(p - intrinsic) < 1e-6f) {
        results[idx] = 0.0f;
        return;
    }

    // Check if price is reasonable (not below intrinsic)
    if (p < intrinsic - 1e-6f) {
        results[idx] = NAN;
        return;
    }

    // Precalculate PV of dividends and Dividend Events (Once per thread/option)
    float pv_divs;
    DivEvent div_events[MAX_DIV_EVENTS];
    int n_div_events;

    precompute_dividends_cuda(t, time_steps, rates_curve, rates_times, n_rates,
                              div_amounts, div_times, n_divs,
                              pv_divs, div_events, n_div_events);

    results[idx] = implied_vol_american_fd_cuda(
        p, s, k, t, is_call,
        rates_curve, rates_times, n_rates,
        pv_divs, div_events, n_div_events,
        tol, max_iter, v_min, v_max,
        time_steps, space_steps);
}

__device__ float price_american_binomial_cash_div_threadlocal(
    float S, float K, float T,
    float sigma, uint8_t is_call, int n_steps,
    const float *rates_curve, const float *rates_times, int n_rates,
    const float *div_amounts, const int *div_steps, int n_divs) {
    constexpr int MAX_STEPS = 512;
    float V[MAX_STEPS + 1];

    // Defensive bound for exponent
    if (n_steps > MAX_STEPS) n_steps = MAX_STEPS;

    // Scaling time steps for short T
    int effective_steps = n_steps;
    if (T < 0.1f) {
        effective_steps = (int)(T * n_steps * 10);
        if (effective_steps > n_steps) effective_steps = n_steps;
        if (effective_steps < 10)       effective_steps = 10;
    }
    if (effective_steps > MAX_STEPS) effective_steps = MAX_STEPS;
    float dt = T / effective_steps;

    if (dt <= 0.0f) {
        return is_call ? fmaxf(S - K, 0.0f) : fmaxf(K - S, 0.0f);
    }
    if (dt < 1e-6f) {
        return is_call ? fmaxf(S - K, 0.0f) : fmaxf(K - S, 0.0f);
    }

    // Standard CRR multipliers (no escrowing)
    float u = expf(sigma * sqrtf(dt));
    if (u <= 1.001f) u = 1.001f;  // numerical safety
    float d = 1.0f / u;
    int   N = effective_steps;

    // 1. Terminal payoffs with cash dividends applied along each path
    float u2      = u * u;
    float curr_S  = S * powf(d, N);  // bottom node (all downs)

    for (int j = 0; j <= N; ++j) {
        // Start from tree stock at (N, j) ignoring dividends
        float S_j = curr_S;

        // Apply each dividend along this path:
        // dividend at step k reduces S by D_k * u^(ups_after_k) * d^(downs_after_k)
        if (n_divs > 0) {
            for (int div_i = 0; div_i < n_divs; ++div_i) {
                int step_div = div_steps[div_i];
                if (step_div > 0 && step_div <= N) {
                    // Ups/downs before the dividend on this path
                    int ups_before_div   = (step_div <= j) ? step_div : j;
                    int downs_before_div = step_div - ups_before_div;

                    int ups_after_div   = j - ups_before_div;
                    int downs_after_div = (N - j) - downs_before_div;

                    float mult_after = powf(u, ups_after_div) * powf(d, downs_after_div);
                    S_j -= div_amounts[div_i] * mult_after;
                    if (S_j < 1e-8f) S_j = 1e-8f;
                }
            }
        }

        V[j] = is_call ? fmaxf(S_j - K, 0.0f) : fmaxf(K - S_j, 0.0f);
        curr_S *= u2;  // move to next node: replace one d with one u
    }

    // 2. Backward induction with term structure and discrete dividends
    for (int step = N - 1; step >= 0; --step) {
        float t_curr = step * dt;
        float r      = get_rate_at_time(t_curr, rates_curve, rates_times, n_rates);
        float disc   = expf(-r * dt);
        float edtr   = expf(r * dt);
        float p      = (edtr - d) / (u - d);
        // clamp to [0,1] to avoid explosions
        p = fmaxf(0.0f, fminf(1.0f, p));

        for (int j = 0; j <= step; ++j) {
            float cont = disc * (p * V[j + 1] + (1.0f - p) * V[j]);

            // Rebuild stock at node (step, j) ignoring dividends
            float S_j = S * powf(u, j) * powf(d, step - j);

            // Apply all dividends that have occurred up to and including this step
            if (n_divs > 0) {
                for (int div_i = 0; div_i < n_divs; ++div_i) {
                    int step_div = div_steps[div_i];
                    if (step_div <= step && step_div > 0) {
                        int ups_before_div   = (step_div <= j) ? step_div : j;
                        int downs_before_div = step_div - ups_before_div;

                        int ups_after_div   = j - ups_before_div;
                        int downs_after_div = (step - j) - downs_before_div;

                        float mult_after = powf(u, ups_after_div) * powf(d, downs_after_div);
                        S_j -= div_amounts[div_i] * mult_after;
                        if (S_j < 1e-8f) S_j = 1e-8f;
                    }
                }
            }

            float ex = is_call ? fmaxf(S_j - K, 0.0f) : fmaxf(K - S_j, 0.0f);
            V[j] = fmaxf(cont, ex);
        }
    }
    return V[0];
}

__device__ float implied_vol_american_bisection_threadlocal(
    float target, float S, float K, float T, uint8_t is_call, int n_steps,
    const float *rates_curve, const float *rates_times, int n_rates,
    const float *div_amounts, const int *div_steps, int n_divs,
    float tol = 1e-6f, int max_iter = 100, float v_min = 1e-4f, float v_max = 5.0f) {
    float p_low = price_american_binomial_cash_div_threadlocal(S, K, T, v_min, is_call, n_steps, rates_curve,
                                                               rates_times, n_rates, div_amounts, div_steps, n_divs);
    float p_high = price_american_binomial_cash_div_threadlocal(S, K, T, v_max, is_call, n_steps, rates_curve,
                                                                rates_times, n_rates, div_amounts, div_steps, n_divs);
    if (target <= p_low + 1e-12f) return 0.0f;
    if (target > p_high + 1e-12f) return NAN;

    float lo = v_min, hi = v_max, mid = 0.0f;
    for (int i = 0; i < max_iter; ++i) {
        mid = 0.5f * (lo + hi);
        float p_mid = price_american_binomial_cash_div_threadlocal(S, K, T, mid, is_call, n_steps, rates_curve,
                                                                   rates_times, n_rates, div_amounts, div_steps,
                                                                   n_divs);
        float diff = p_mid - target;
        if (fabsf(diff) < tol) return mid;
        if (diff < 0.0f) lo = mid;
        else hi = mid;
        if (hi - lo < tol) break;
    }
    return mid;
}

__device__ float american_binomial_delta_fd(
    float S, float K, float T, float sigma, uint8_t is_call, int n_steps,
    const float *rates_curve, const float *rates_times, int n_rates,
    const float *div_amounts, const int *div_steps, int n_divs,
    float rel_shift = 1e-4f) {
    float h = S * rel_shift;
    if (h < 1e-6f) h = 1e-6f;
    float up = price_american_binomial_cash_div_threadlocal(
        S + h, K, T, sigma, is_call, n_steps, rates_curve, rates_times, n_rates, div_amounts, div_steps, n_divs);
    float down = price_american_binomial_cash_div_threadlocal(
        S - h, K, T, sigma, is_call, n_steps, rates_curve, rates_times, n_rates, div_amounts, div_steps, n_divs);
    return (up - down) / (2.0f * h);
}

__device__ float american_binomial_vega_fd(
    float S, float K, float T, float sigma, uint8_t is_call, int n_steps,
    const float *rates_curve, const float *rates_times, int n_rates,
    const float *div_amounts, const int *div_steps, int n_divs,
    float abs_shift = 1e-4f) {
    float h = fmaxf(abs_shift, sigma * 1e-2f); // Robust shift for high vols
    float up = price_american_binomial_cash_div_threadlocal(
        S, K, T, sigma + h, is_call, n_steps, rates_curve, rates_times, n_rates, div_amounts, div_steps, n_divs);
    float down = price_american_binomial_cash_div_threadlocal(
        S, K, T, sigma - h, is_call, n_steps, rates_curve, rates_times, n_rates, div_amounts, div_steps, n_divs);
    return (up - down) / (2.0f * h);
}

// =======================
// Main IV kernel: no shared memory needed
// =======================

__global__ void compute_iv_kernel_threadlocal(
    const float *prices, const float *spots, const float *strikes, const float *tenors,
    const uint8_t *v_is_call, const float *rates_curve, const float *rates_times, int n_rates,
    int n_options,
    const float *div_amounts, const float *div_times, int n_divs,
    float *results,
    float tol = 1e-6f, int max_iter = 100, float v_min = 1e-4f, float v_max = 5.0f, float steps_factor = 1.0f) {
    int n_steps = 200 * steps_factor;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_options) return;

    float p = prices[idx];
    float s = spots[idx];
    float k = strikes[idx];
    float t = tenors[idx];
    uint8_t is_call = v_is_call[idx];

    if (!isfinite(p) || !isfinite(s) || !isfinite(k) || !isfinite(t) ||
        t <= 0.0f || s <= 0.0f || k <= 0.0f || p <= 0.0f) {
        results[idx] = (t <= 0.0f) ? 0.0f : NAN;
        return;
    }

    // Only include dividends before expiry
    const int MAX_DIVS = 32;
    float local_div_amounts[MAX_DIVS];
    int local_div_steps[MAX_DIVS];
    int valid_divs = 0;

    float intrinsic = is_call ? max(s - k, 0.0f) : max(k - s, 0.0f);
    if (fabs(p - intrinsic) < 1e-6f) {
        results[idx] = 0.0f;
        return;
    }

    for (int j = 0; j < n_divs && j < MAX_DIVS; ++j) {
        float div_time = div_times[j];
        if (div_time > 0.0f && div_time < t) {
            int step = (int) roundf(div_time / t * n_steps);
            step = max(1, min(step, n_steps));
            local_div_steps[valid_divs] = step;
            local_div_amounts[valid_divs] = div_amounts[j];
            valid_divs++;
        }
    }

    results[idx] = implied_vol_american_bisection_threadlocal(
        p, s, k, t, is_call, n_steps,
        rates_curve, rates_times, n_rates,
        local_div_amounts, local_div_steps, valid_divs,
        tol, max_iter, v_min, v_max
    );
}

__global__ void compute_price_kernel_threadlocal(
    const float *spots, const float *strikes, const float *tenors, const float *sigmas,
    const uint8_t *rights,
    const float *rates_curve, const float *rates_times, int n_rates,
    const float *div_amounts, const float *div_times, int n_divs,
    int time_steps,
    int space_steps,
    int n_options,
    float *out_price) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_options) return;

    float s = spots[idx];
    float k = strikes[idx];
    float t = tenors[idx];
    float sigma = sigmas[idx];
    int right = rights[idx];

    if (!isfinite(s) || !isfinite(k) || !isfinite(t) || !isfinite(sigma) ||
        t <= 0.0f || s <= 0.0f || k <= 0.0f || sigma <= 0.0f) {
        out_price[idx] = NAN;
        return;
    }

    // Precalculate PV of dividends and Dividend Events (Once per thread/option)
    float pv_divs;
    DivEvent div_events[MAX_DIV_EVENTS];
    int n_div_events;

    precompute_dividends_cuda(t, time_steps, rates_curve, rates_times, n_rates,
                              div_amounts, div_times, n_divs,
                              pv_divs, div_events, n_div_events);

    out_price[idx] = price_american_fd_div_cuda(
        s, k, t, sigma, right,
        rates_curve, rates_times, n_rates,
        pv_divs, div_events, n_div_events,
        time_steps, space_steps
    );
}

__global__ void compute_delta_kernel_threadlocal(
    const float *spots, const float *strikes, const float *tenors, const float *sigmas,
    const int *rights, int n_steps, int n_options,
    const float *rates_curve, const float *rates_times, int n_rates,
    const float *div_amounts, const float *div_times, int n_divs,
    float *out_delta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_options) return;

    float s = spots[idx];
    float k = strikes[idx];
    float t = tenors[idx];
    float sigma = sigmas[idx];
    int right = rights[idx];

    if (!isfinite(s) || !isfinite(k) || !isfinite(t) || !isfinite(sigma) ||
        t <= 0.0f || s <= 0.0f || k <= 0.0f || sigma <= 0.0f) {
        out_delta[idx] = NAN;
        return;
    }

    const int MAX_DIVS = 32;
    float local_div_amounts[MAX_DIVS];
    int local_div_steps[MAX_DIVS];
    int valid_divs = 0;
    for (int j = 0; j < n_divs && j < MAX_DIVS; ++j) {
        float div_time = div_times[j];
        if (div_time > 0.0f && div_time < t) {
            int step = (int) roundf(div_time / t * n_steps);
            step = max(1, min(step, n_steps));
            local_div_steps[valid_divs] = step;
            local_div_amounts[valid_divs] = div_amounts[j];
            valid_divs++;
        }
    }

    out_delta[idx] = american_binomial_delta_fd(
        s, k, t, sigma, right, n_steps,
        rates_curve, rates_times, n_rates,
        local_div_amounts, local_div_steps, valid_divs
    );
}

__global__ void compute_vega_kernel_threadlocal(
    const float *spots, const float *strikes, const float *tenors, const float *sigmas,
    const uint8_t *v_is_call, int n_steps, int n_options,
    const float *rates_curve, const float *rates_times, int n_rates,
    const float *div_amounts, const float *div_times, int n_divs,
    float *out_vega) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_options) return;

    float s = spots[idx];
    float k = strikes[idx];
    float t = tenors[idx];
    float sigma = sigmas[idx];
    uint8_t is_call = v_is_call[idx];

    if (!isfinite(s) || !isfinite(k) || !isfinite(t) || !isfinite(sigma) ||
        t <= 0.0f || s <= 0.0f || k <= 0.0f || sigma <= 0.0f) {
        out_vega[idx] = NAN;
        return;
    }

    const int MAX_DIVS = 32;
    float local_div_amounts[MAX_DIVS];
    int local_div_steps[MAX_DIVS];
    int valid_divs = 0;
    for (int j = 0; j < n_divs && j < MAX_DIVS; ++j) {
        float div_time = div_times[j];
        if (div_time > 0.0f && div_time < t) {
            int step = (int) roundf(div_time / t * n_steps);
            step = max(1, min(step, n_steps));
            local_div_steps[valid_divs] = step;
            local_div_amounts[valid_divs] = div_amounts[j];
            valid_divs++;
        }
    }

    out_vega[idx] = american_binomial_vega_fd(
        s, k, t, sigma, is_call, n_steps,
        rates_curve, rates_times, n_rates,
        local_div_amounts, local_div_steps, valid_divs
    );
}

/////////////////////////////////////////////////////////
/// Host / CPU wrappers that launch kernels
/////////////////////////////////////////////////////////

// Host wrapper for FD IV calculation on GPU - refactored for accuracy
std::vector<float> get_v_iv_fd_cuda_new(
    const std::vector<float> &prices,
    const std::vector<float> &spots,
    const std::vector<float> &strikes,
    const std::vector<float> &tenors,
    const std::vector<uint8_t> &v_is_call,
    const std::vector<float> &rates_curve,
    const std::vector<float> &rates_times,
    const std::vector<float> &div_amounts,
    const std::vector<float> &div_times,
    const float tol,
    const int max_iter,
    const float v_min,
    const float v_max,
    const int time_steps,
    const int space_steps
) {
    int n_options = prices.size();
    std::vector<float> results(n_options, std::numeric_limits<float>::quiet_NaN());

    if (n_options == 0) return results;

    // Validate input sizes
    if (spots.size() != n_options || strikes.size() != n_options ||
        tenors.size() != n_options || v_is_call.size() != n_options) {
        return results;
    }

    int n_divs = div_amounts.size();
    int n_rates = rates_curve.size();

    // Validate rate and dividend data
    if (!rates_curve.empty() && rates_curve.size() != rates_times.size()) {
        return results;
    }
    if (!div_amounts.empty() && div_amounts.size() != div_times.size()) {
        return results;
    }

    float *d_prices, *d_spots, *d_strikes, *d_tenors, *d_results;
    uint8_t *d_v_is_call;
    float *d_div_amounts = nullptr, *d_div_times = nullptr;
    float *d_rates_curve = nullptr, *d_rates_times = nullptr;

    cudaMalloc(&d_prices, n_options * sizeof(float));
    cudaMalloc(&d_spots, n_options * sizeof(float));
    cudaMalloc(&d_strikes, n_options * sizeof(float));
    cudaMalloc(&d_tenors, n_options * sizeof(float));
    cudaMalloc(&d_v_is_call, n_options * sizeof(uint8_t));
    cudaMalloc(&d_results, n_options * sizeof(float));

    cudaMemcpy(d_prices, prices.data(), n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_spots, spots.data(), n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_strikes, strikes.data(), n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tenors, tenors.data(), n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_is_call, v_is_call.data(), n_options * sizeof(uint8_t), cudaMemcpyHostToDevice);

    if (n_divs > 0) {
        cudaMalloc(&d_div_amounts, n_divs * sizeof(float));
        cudaMalloc(&d_div_times, n_divs * sizeof(float));
        cudaMemcpy(d_div_amounts, div_amounts.data(), n_divs * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_div_times, div_times.data(), n_divs * sizeof(float), cudaMemcpyHostToDevice);
    }

    if (n_rates > 0) {
        cudaMalloc(&d_rates_curve, n_rates * sizeof(float));
        cudaMalloc(&d_rates_times, n_rates * sizeof(float));
        cudaMemcpy(d_rates_curve, rates_curve.data(), n_rates * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_rates_times, rates_times.data(), n_rates * sizeof(float), cudaMemcpyHostToDevice);
    }

    int block_size = 256;
    int grid_size = (n_options + block_size - 1) / block_size;

    compute_fd_iv_kernel<<<grid_size, block_size>>>(
        d_prices, d_spots, d_strikes, d_tenors, d_v_is_call, n_options,
        d_rates_curve, d_rates_times, n_rates,
        d_div_amounts, d_div_times, n_divs, d_results,
        tol, max_iter, v_min, v_max, time_steps, space_steps
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_prices);
        cudaFree(d_spots);
        cudaFree(d_strikes);
        cudaFree(d_tenors);
        cudaFree(d_v_is_call);
        cudaFree(d_results);
        if (d_div_amounts) cudaFree(d_div_amounts);
        if (d_div_times) cudaFree(d_div_times);
        if (d_rates_curve) cudaFree(d_rates_curve);
        if (d_rates_times) cudaFree(d_rates_times);
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }

    cudaDeviceSynchronize();

    cudaMemcpy(results.data(), d_results, n_options * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_prices);
    cudaFree(d_spots);
    cudaFree(d_strikes);
    cudaFree(d_tenors);
    cudaFree(d_v_is_call);
    cudaFree(d_results);
    if (d_div_amounts) cudaFree(d_div_amounts);
    if (d_div_times) cudaFree(d_div_times);
    if (d_rates_curve) cudaFree(d_rates_curve);
    if (d_rates_times) cudaFree(d_rates_times);

    return results;
}

std::vector<float> get_v_fd_price_cuda(
    const std::vector<float> &spots,
    const std::vector<float> &strikes,
    const std::vector<float> &tenors,
    const std::vector<float> &sigmas,
    const std::vector<uint8_t> &v_is_call,
    const std::vector<float> &rates_curve,
    const std::vector<float> &rates_times,
    const std::vector<float> &div_amounts,
    const std::vector<float> &div_times,
    const int time_steps,
    const int space_steps
) {
    int n_options = spots.size();
    std::vector<float> prices(n_options);

    if (n_options == 0) return prices;

    int n_divs = div_amounts.size();
    int n_rates = rates_curve.size();

    float *d_spots, *d_strikes, *d_tenors, *d_sigmas;
    uint8_t *d_v_is_call;
    float *d_div_amounts = nullptr, *d_div_times = nullptr;
    float *d_prices;
    float *d_rates_curve = nullptr, *d_rates_times = nullptr;

    cudaMalloc(&d_spots, n_options * sizeof(float));
    cudaMalloc(&d_strikes, n_options * sizeof(float));
    cudaMalloc(&d_tenors, n_options * sizeof(float));
    cudaMalloc(&d_sigmas, n_options * sizeof(float));
    cudaMalloc(&d_v_is_call, n_options * sizeof(uint8_t));
    cudaMalloc(&d_prices, n_options * sizeof(float));

    cudaMemcpy(d_spots, spots.data(), n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_strikes, strikes.data(), n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tenors, tenors.data(), n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sigmas, sigmas.data(), n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_is_call, v_is_call.data(), n_options * sizeof(uint8_t), cudaMemcpyHostToDevice);

    if (n_divs > 0 && div_amounts.size() == div_times.size()) {
        cudaMalloc(&d_div_amounts, n_divs * sizeof(float));
        cudaMalloc(&d_div_times, n_divs * sizeof(float));
        cudaMemcpy(d_div_amounts, div_amounts.data(), n_divs * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_div_times, div_times.data(), n_divs * sizeof(float), cudaMemcpyHostToDevice);
    }

    if (n_rates > 0 && rates_curve.size() == rates_times.size()) {
        cudaMalloc(&d_rates_curve, n_rates * sizeof(float));
        cudaMalloc(&d_rates_times, n_rates * sizeof(float));
        cudaMemcpy(d_rates_curve, rates_curve.data(), n_rates * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_rates_times, rates_times.data(), n_rates * sizeof(float), cudaMemcpyHostToDevice);
    }

    int block_size = 256;
    int grid_size = (n_options + block_size - 1) / block_size;

    compute_price_kernel_threadlocal<<<grid_size, block_size>>>(
        d_spots, d_strikes, d_tenors, d_sigmas, d_v_is_call,
        d_rates_curve, d_rates_times, n_rates,
        d_div_amounts, d_div_times, n_divs,
        time_steps, space_steps, n_options, d_prices
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));

    cudaMemcpy(prices.data(), d_prices, n_options * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_spots);
    cudaFree(d_strikes);
    cudaFree(d_tenors);
    cudaFree(d_sigmas);
    cudaFree(d_v_is_call);
    cudaFree(d_prices);
    if (d_div_amounts) cudaFree(d_div_amounts);
    if (d_div_times) cudaFree(d_div_times);
    if (d_rates_curve) cudaFree(d_rates_curve);
    if (d_rates_times) cudaFree(d_rates_times);

    return prices;
}

std::vector<float> get_v_fd_delta_cuda(
    const std::vector<float> &spots,
    const std::vector<float> &strikes,
    const std::vector<float> &tenors,
    const std::vector<uint8_t> &v_is_call,
    const std::vector<float> &sigmas,
    int n_steps,
    const std::vector<float> &rates_curve,
    const std::vector<float> &rates_times,
    const std::vector<float> &div_amounts,
    const std::vector<float> &div_times) {
    int n_options = spots.size();
    std::vector<float> deltas(n_options);

    if (n_options == 0) return deltas;

    int n_divs = div_amounts.size();
    int n_rates = rates_curve.size();

    float *d_spots, *d_strikes, *d_tenors, *d_sigmas;
    int *d_v_is_call;
    float *d_div_amounts = nullptr, *d_div_times = nullptr;
    float *d_deltas;
    float *d_rates_curve = nullptr, *d_rates_times = nullptr;

    cudaMalloc(&d_spots, n_options * sizeof(float));
    cudaMalloc(&d_strikes, n_options * sizeof(float));
    cudaMalloc(&d_tenors, n_options * sizeof(float));
    cudaMalloc(&d_sigmas, n_options * sizeof(float));
    cudaMalloc(&d_v_is_call, n_options * sizeof(uint8_t));
    cudaMalloc(&d_deltas, n_options * sizeof(float));

    cudaMemcpy(d_spots, spots.data(), n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_strikes, strikes.data(), n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tenors, tenors.data(), n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sigmas, sigmas.data(), n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_is_call, v_is_call.data(), n_options * sizeof(uint8_t), cudaMemcpyHostToDevice);

    if (n_divs > 0 && div_amounts.size() == div_times.size()) {
        cudaMalloc(&d_div_amounts, n_divs * sizeof(float));
        cudaMalloc(&d_div_times, n_divs * sizeof(float));
        cudaMemcpy(d_div_amounts, div_amounts.data(), n_divs * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_div_times, div_times.data(), n_divs * sizeof(float), cudaMemcpyHostToDevice);
    }

    if (n_rates > 0 && rates_curve.size() == rates_times.size()) {
        cudaMalloc(&d_rates_curve, n_rates * sizeof(float));
        cudaMalloc(&d_rates_times, n_rates * sizeof(float));
        cudaMemcpy(d_rates_curve, rates_curve.data(), n_rates * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_rates_times, rates_times.data(), n_rates * sizeof(float), cudaMemcpyHostToDevice);
    }

    int block_size = 256;
    int grid_size = (n_options + block_size - 1) / block_size;

    compute_delta_kernel_threadlocal<<<grid_size, block_size>>>(
        d_spots, d_strikes, d_tenors, d_sigmas, d_v_is_call, n_steps, n_options,
        d_rates_curve, d_rates_times, n_rates,
        d_div_amounts, d_div_times, n_divs, d_deltas
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));

    cudaMemcpy(deltas.data(), d_deltas, n_options * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_spots);
    cudaFree(d_strikes);
    cudaFree(d_tenors);
    cudaFree(d_sigmas);
    cudaFree(d_v_is_call);
    cudaFree(d_deltas);
    if (d_div_amounts) cudaFree(d_div_amounts);
    if (d_div_times) cudaFree(d_div_times);
    if (d_rates_curve) cudaFree(d_rates_curve);
    if (d_rates_times) cudaFree(d_rates_times);

    return deltas;
}

std::vector<float> get_v_fd_vega_cuda(
    const std::vector<float> &spots,
    const std::vector<float> &strikes,
    const std::vector<float> &tenors,
    const std::vector<uint8_t> &v_is_call,
    const std::vector<float> &sigmas,
    int n_steps,
    const std::vector<float> &rates_curve,
    const std::vector<float> &rates_times,
    const std::vector<float> &div_amounts,
    const std::vector<float> &div_times) {
    int n_options = spots.size();
    std::vector<float> vegas(n_options);

    if (n_options == 0) return vegas;

    int n_divs = div_amounts.size();
    int n_rates = rates_curve.size();

    float *d_spots, *d_strikes, *d_tenors, *d_sigmas;
    uint8_t *d_v_is_call;
    float *d_div_amounts = nullptr, *d_div_times = nullptr;
    float *d_vegas;
    float *d_rates_curve = nullptr, *d_rates_times = nullptr;

    cudaMalloc(&d_spots, n_options * sizeof(float));
    cudaMalloc(&d_strikes, n_options * sizeof(float));
    cudaMalloc(&d_tenors, n_options * sizeof(float));
    cudaMalloc(&d_sigmas, n_options * sizeof(float));
    cudaMalloc(&d_v_is_call, n_options * sizeof(uint8_t));
    cudaMalloc(&d_vegas, n_options * sizeof(float));

    cudaMemcpy(d_spots, spots.data(), n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_strikes, strikes.data(), n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tenors, tenors.data(), n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sigmas, sigmas.data(), n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_is_call, v_is_call.data(), n_options * sizeof(uint8_t), cudaMemcpyHostToDevice);

    if (n_divs > 0 && div_amounts.size() == div_times.size()) {
        cudaMalloc(&d_div_amounts, n_divs * sizeof(float));
        cudaMalloc(&d_div_times, n_divs * sizeof(float));
        cudaMemcpy(d_div_amounts, div_amounts.data(), n_divs * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_div_times, div_times.data(), n_divs * sizeof(float), cudaMemcpyHostToDevice);
    }

    if (n_rates > 0 && rates_curve.size() == rates_times.size()) {
        cudaMalloc(&d_rates_curve, n_rates * sizeof(float));
        cudaMalloc(&d_rates_times, n_rates * sizeof(float));
        cudaMemcpy(d_rates_curve, rates_curve.data(), n_rates * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_rates_times, rates_times.data(), n_rates * sizeof(float), cudaMemcpyHostToDevice);
    }

    int block_size = 256;
    int grid_size = (n_options + block_size - 1) / block_size;

    compute_vega_kernel_threadlocal<<<grid_size, block_size>>>(
        d_spots, d_strikes, d_tenors, d_sigmas, d_v_is_call, n_steps, n_options,
        d_rates_curve, d_rates_times, n_rates,
        d_div_amounts, d_div_times, n_divs, d_vegas
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));

    cudaMemcpy(vegas.data(), d_vegas, n_options * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_spots);
    cudaFree(d_strikes);
    cudaFree(d_tenors);
    cudaFree(d_sigmas);
    cudaFree(d_v_is_call);
    cudaFree(d_vegas);
    if (d_div_amounts) cudaFree(d_div_amounts);
    if (d_div_times) cudaFree(d_div_times);
    if (d_rates_curve) cudaFree(d_rates_curve);
    if (d_rates_times) cudaFree(d_rates_times);

    return vegas;
}

std::vector<float> get_v_iv_binomial_cuda(
    const std::vector<float> &prices,
    const std::vector<float> &spots,
    const std::vector<float> &strikes,
    const std::vector<float> &tenors,
    const std::vector<uint8_t> &v_is_call,
    const std::vector<float> &rates_curve,
    const std::vector<float> &rates_times,
    const std::vector<float> &div_amounts,
    const std::vector<float> &div_times,
    const float tol,
    const int max_iter,
    const float v_min,
    const float v_max,
    const float steps_factor
) {
    int n_options = prices.size();
    std::vector<float> results(n_options);

    if (n_options == 0) return results;

    int n_divs = div_amounts.size();
    int n_rates = rates_curve.size();

    // Allocate device memory (inputs + outputs)
    float *d_prices, *d_spots, *d_strikes, *d_tenors, *d_results;
    uint8_t *d_v_is_call;
    float *d_div_amounts = nullptr, *d_div_times = nullptr;
    float *d_rates_curve = nullptr, *d_rates_times = nullptr;

    cudaMalloc(&d_prices, n_options * sizeof(float));
    cudaMalloc(&d_spots, n_options * sizeof(float));
    cudaMalloc(&d_strikes, n_options * sizeof(float));
    cudaMalloc(&d_tenors, n_options * sizeof(float));
    cudaMalloc(&d_v_is_call, n_options * sizeof(uint8_t));
    cudaMalloc(&d_results, n_options * sizeof(float));

    cudaMemcpy(d_prices, prices.data(), n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_spots, spots.data(), n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_strikes, strikes.data(), n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tenors, tenors.data(), n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_is_call, v_is_call.data(), n_options * sizeof(uint8_t), cudaMemcpyHostToDevice);

    if (n_divs > 0 && div_amounts.size() == div_times.size()) {
        cudaMalloc(&d_div_amounts, n_divs * sizeof(float));
        cudaMalloc(&d_div_times, n_divs * sizeof(float));
        cudaMemcpy(d_div_amounts, div_amounts.data(), n_divs * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_div_times, div_times.data(), n_divs * sizeof(float), cudaMemcpyHostToDevice);
    }

    if (n_rates > 0 && rates_curve.size() == rates_times.size()) {
        cudaMalloc(&d_rates_curve, n_rates * sizeof(float));
        cudaMalloc(&d_rates_times, n_rates * sizeof(float));
        cudaMemcpy(d_rates_curve, rates_curve.data(), n_rates * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_rates_times, rates_times.data(), n_rates * sizeof(float), cudaMemcpyHostToDevice);
    }

    // Launch with no shared_mem needed
    int block_size = 256;
    int grid_size = (n_options + block_size - 1) / block_size;

    compute_iv_kernel_threadlocal<<<grid_size, block_size>>>(
        d_prices, d_spots, d_strikes, d_tenors, d_v_is_call,
        d_rates_curve, d_rates_times, n_rates,
        n_options,
        d_div_amounts, d_div_times, n_divs, d_results,
        tol, max_iter, v_min, v_max, steps_factor
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));

    cudaMemcpy(results.data(), d_results, n_options * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_prices);
    cudaFree(d_spots);
    cudaFree(d_strikes);
    cudaFree(d_tenors);
    cudaFree(d_v_is_call);
    cudaFree(d_results);
    if (d_div_amounts) cudaFree(d_div_amounts);
    if (d_div_times) cudaFree(d_div_times);
    if (d_rates_curve) cudaFree(d_rates_curve);
    if (d_rates_times) cudaFree(d_rates_times);

    return results;
}