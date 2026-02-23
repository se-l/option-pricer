#include "pricing_engine.h"

#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdexcept>
#include <unordered_map>


// --- NEW: precomputed per-tenor rate/discount schedules (to avoid curve interpolation in FD loop) ---
constexpr int MAX_TIME_STEPS_SCHED = 256;
constexpr int RATE_SCHED_STRIDE = MAX_TIME_STEPS_SCHED + 1; // store [0..N_t] inclusive

// Device version of tridiagonal solver (Thomas algorithm)
__device__ inline void solve_tridiagonal_cuda(
    int n,
    const float* a,  // sub-diagonal [1..n-1]
    const float* b,  // diagonal    [0..n-1]
    const float* c,  // super-diag  [0..n-2]
    float* d)        // RHS         [0..n-1], overwritten by solution
{
    // Use stack arrays for temporaries
    constexpr int MAX_GRID = 257;
    float c_star[MAX_GRID];
    float d_star[MAX_GRID];

    if (n > MAX_GRID) return; // Safety check (but now matches FD cap)

    c_star[0] = c[0] / b[0];
    d_star[0] = d[0] / b[0];

    for (int i = 1; i < n; ++i) {
        float denom = (b[i] - a[i] * c_star[i - 1]);

        // Prevent division by (near-)zero leading to inf/NaN
        if (fabsf(denom) < 1e-20f) denom = (denom < 0.0f ? -1e-20f : 1e-20f);

        float m = 1.0f / denom;
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

// --- NEW: Monotone-convex style instantaneous forward from zero curve R(0,t) ---
__device__ inline float get_rate_at_time_convex_monotone(
    float t,
    const float* rates_curve,
    const float* rates_times,
    int n_points)
{
    if (n_points <= 0) return 0.0f;
    if (t <= 0.0f) return rates_curve[0];

    if (t <= rates_times[0]) return rates_curve[0];
    if (t >= rates_times[n_points - 1]) return rates_curve[n_points - 1];

    int k = 0;
    for (int i = 0; i < n_points - 1; ++i) {
        if (t <= rates_times[i + 1]) { k = i; break; }
    }

    auto Z = [&](int i) -> float { return rates_curve[i] * rates_times[i]; };
    auto f_interval = [&](int i) -> float {
        float t0 = rates_times[i];
        float t1 = rates_times[i + 1];
        float h  = t1 - t0;
        if (!(h > 0.0f)) return 0.0f;
        return (Z(i + 1) - Z(i)) / h;
    };
    auto clampf = [&](float x, float lo, float hi) -> float {
        return fminf(fmaxf(x, lo), hi);
    };

    auto fhat = [&](int i) -> float {
        if (i <= 0) return f_interval(0);
        if (i >= n_points - 1) return f_interval(n_points - 2);

        float fL = f_interval(i - 1);
        float fR = f_interval(i);

        if (fL * fR <= 0.0f) return 0.0f;

        float hL = rates_times[i] - rates_times[i - 1];
        float hR = rates_times[i + 1] - rates_times[i];
        float wL = 2.0f * hR + hL;
        float wR = hR + 2.0f * hL;

        float denom = (wL / fL) + (wR / fR);
        if (fabsf(denom) < 1e-20f) return 0.0f;
        float fh = (wL + wR) / denom;

        float lo = fminf(fL, fR);
        float hi = fmaxf(fL, fR);
        return clampf(fh, lo, hi);
    };

    float t0 = rates_times[k];
    float t1 = rates_times[k + 1];
    float h  = t1 - t0;
    if (!(h > 0.0f)) return rates_curve[k];

    float x    = (t - t0) / h;
    float fbar = f_interval(k);
    float fL   = fhat(k);
    float fR   = fhat(k + 1);

    float c = 6.0f * (fbar - 0.5f * (fL + fR));

    float f_mid = 0.5f * (fL + fR) + 0.25f * c;
    float lo = fminf(fL, fR);
    float hi = fmaxf(fL, fR);
    if (f_mid < lo || f_mid > hi) {
        float target = (f_mid < lo) ? lo : hi;
        c = 4.0f * (target - 0.5f * (fL + fR));
    }

    float f = fL + (fR - fL) * x + c * x * (1.0f - x);
    f = clampf(f, lo - 1e6f, hi + 1e6f);
    if (!isfinite(f)) return rates_curve[k];
    return f;
}

__device__ inline float get_Z_at_time(float t, const float* rates_curve,
                                      const float* rates_times, int n_points) {
    // Z(t) = R(t) * t with linear interpolation on (time, Z) points.
    if (n_points <= 0) return 0.0f;
    if (t <= 0.0f) return 0.0f;

    if (t <= rates_times[0]) {
        return rates_curve[0] * t;
    }
    if (t >= rates_times[n_points - 1]) {
        return rates_curve[n_points - 1] * t;
    }

    for (int i = 0; i < n_points - 1; ++i) {
        if (t <= rates_times[i + 1]) {
            float t0 = rates_times[i];
            float t1 = rates_times[i + 1];
            float z0 = rates_curve[i] * t0;
            float z1 = rates_curve[i + 1] * t1;

            float w = (t - t0) / (t1 - t0);
            return (1.0f - w) * z0 + w * z1;
        }
    }
    return rates_curve[n_points - 1] * t;
}

__device__ inline float df_0_t(float t, const float* rates_curve,
                               const float* rates_times, int n_points) {
    return expf(-get_Z_at_time(t, rates_curve, rates_times, n_points));
}

__device__ inline float df_t_T(float t, float T, const float* rates_curve,
                               const float* rates_times, int n_points) {
    float Zt = get_Z_at_time(t, rates_curve, rates_times, n_points);
    float ZT = get_Z_at_time(T, rates_curve, rates_times, n_points);
    return expf(-(ZT - Zt));
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
            // Use curve-consistent discount factor D(0,t_div)
            pv_divs += div_amounts[i] * df_0_t(t_div, rates_curve, rates_times, n_rates);
        }
    }
    return pv_divs;
}

// Build dividend schedule
struct DivEvent {
    int step;
    float amount;
    float time;
};
constexpr int MAX_DIV_EVENTS = 32;

__device__ float price_american_fd_cuda(
    const float S, const float K, const float T,
    const float sigma_eval, const uint8_t is_call,
    const float* rates_curve,
    const float* rates_times,
    const int n_rates,
    const float pv_divs,
    const DivEvent* div_events,
    const int n_div_events,
    const int time_steps,
    const int space_steps,
    const float* r_sched,        // NEW: length RATE_SCHED_STRIDE
    const float* df_tT_sched)     // NEW: length RATE_SCHED_STRIDE
{
    constexpr int MAX_TIME_STEPS = 256;
    constexpr int MAX_SPACE_STEPS = 256;

    if (T <= 0.0f || S <= 0.0f || K <= 0.0f || sigma_eval <= 0.0f)
        return is_call ? fmaxf(S - K, 0.0f) : fmaxf(K - S, 0.0f);

    const int N_t = (time_steps < 10) ? 10 : ((time_steps > MAX_TIME_STEPS) ? MAX_TIME_STEPS : time_steps);
    const int N_s = (space_steps < 10) ? 10 : ((space_steps > MAX_SPACE_STEPS) ? MAX_SPACE_STEPS : space_steps);

    const float dt = T / static_cast<float>(N_t);
    // Use precomputed r(0) when available; fall back to curve if schedule not provided.
    const float r0 = (r_sched != nullptr) ? r_sched[0]
        : get_rate_at_time_convex_monotone(0.0f, rates_curve, rates_times, n_rates);

    // IMPORTANT: build the domain using sigma_grid (constant during IV solve)
    constexpr float S_min = 0.0f;
    const float S_max = fmaxf(
        (S + pv_divs) * expf(fminf(r0 * T + 4.0f * sigma_eval * sqrt(T), 80.0f)),  // 80 protects against overflow
        3.0f * fmaxf(S, K)
    );

    const float dS = (S_max - S_min) / static_cast<float>(N_s);

    float S_grid[MAX_SPACE_STEPS + 1];
    for (int i = 0; i <= N_s; ++i) {
        S_grid[i] = S_min + i * dS;
    }

    float V[MAX_SPACE_STEPS + 1];
    float a[MAX_SPACE_STEPS + 1], b[MAX_SPACE_STEPS + 1], c[MAX_SPACE_STEPS + 1];
    float rhs[MAX_SPACE_STEPS + 1];
    float V_new[MAX_SPACE_STEPS + 1];

    for (int i = 0; i <= N_s; ++i) {
        float payoff = is_call ? fmaxf(S_grid[i] - K, 0.0f) : fmaxf(K - S_grid[i], 0.0f);
        V[i] = payoff;
    }

    // Backward time stepping
    for (int n = N_t - 1; n >= 0; --n) {
        const float t = n * dt;

        // Use precomputed instantaneous forward rate r(t) when available.
        const float r = (r_sched != nullptr) ? r_sched[n]
            : get_rate_at_time_convex_monotone(t, rates_curve, rates_times, n_rates);

        const float theta = 0.5f;
        const float sigma2 = sigma_eval * sigma_eval;

        for (int i = 1; i < N_s; ++i) {
            const float Si = S_grid[i];
            if (Si < 1e-8f) {
                a[i] = 0.0f;
                b[i] = 1.0f;
                c[i] = 0.0f;
                rhs[i] = V[i];
                continue;
            }

            const float Si2 = Si * Si;

            const float coef_pp = 0.5f * sigma2 * Si2 / (dS * dS);
            const float coef_p  = 0.5f * r * Si / dS;
            const float coef_0  = -sigma2 * Si2 / (dS * dS) - r;

            const float alpha = coef_pp - coef_p;
            const float beta  = coef_0;
            const float gamma = coef_pp + coef_p;

            a[i] = -theta * dt * alpha;
            b[i] = 1.0f - theta * dt * beta;
            c[i] = -theta * dt * gamma;

            rhs[i] = V[i] * (1.0f + (1.0f - theta) * dt * beta)
                   + V[i-1] * (1.0f - theta) * dt * alpha
                   + V[i+1] * (1.0f - theta) * dt * gamma;
        }

        // Lower boundary: S = 0
        {
            a[0] = 0.0f; b[0] = 1.0f; c[0] = 0.0f;
            const float df = (df_tT_sched != nullptr) ? df_tT_sched[n]
                : df_t_T(t, T, rates_curve, rates_times, n_rates);
            rhs[0] = is_call ? 0.0f : K * df;
        }

        // Upper boundary: S = S_max
        {
            a[N_s] = 0.0f; b[N_s] = 1.0f; c[N_s] = 0.0f;
            const float df = (df_tT_sched != nullptr) ? df_tT_sched[n]
                : df_t_T(t, T, rates_curve, rates_times, n_rates);
            rhs[N_s] = is_call ? (S_grid[N_s] - K * df) : 0.0f;
        }

        solve_tridiagonal_cuda(N_s + 1, a, b, c, rhs);
        for (int i = 0; i <= N_s; ++i) V[i] = rhs[i];

        // American early exercise at time t (before dividend)
        for (int i = 0; i <= N_s; ++i) {
            const float intrinsic = is_call ? fmaxf(S_grid[i] - K, 0.0f) : fmaxf(K - S_grid[i], 0.0f);
            V[i] = fmaxf(V[i], intrinsic);
        }

        // Apply dividend jumps
        for (int div_idx = 0; div_idx < n_div_events; ++div_idx) {
            if (div_events[div_idx].step == n) {
                for (int i = 0; i <= N_s; ++i) {
                    const float S_pre_div  = S_grid[i];
                    const float S_post_div = S_pre_div - div_events[div_idx].amount;

                    if (S_post_div <= 0.0f) {
                        // Stock worthless after dividend
                        V_new[i] = is_call ? 0.0f : K;
                    } else if (S_post_div >= S_max) {
                        V_new[i] = V[N_s];
                    } else if (S_post_div <= S_min) {
                        V_new[i] = V[0];
                    } else {
                        // Linear interpolation
                        const float idx_f = (S_post_div - S_min) / dS;
                        int idx = static_cast<int>(floorf(idx_f));
                        if (idx < 0) idx = 0;
                        if (idx >= N_s) idx = N_s - 1;
                        const float w = idx_f - idx;
                        V_new[i] = (1.0f - w) * V[idx] + w * V[idx + 1];
                    }
                }
                // Copy back
                for (int i = 0; i <= N_s; ++i) V[i] = V_new[i];

                // Re-apply early exercise after dividend
                for (int i = 0; i <= N_s; ++i) {
                    const float intrinsic = is_call ? fmaxf(S_grid[i] - K, 0.0f) : fmaxf(K - S_grid[i], 0.0f);
                    V[i] = fmaxf(V[i], intrinsic);
                }
            }
        }
    }

    // Final interpolation at spot S
    if (S <= S_min) return V[0];
    if (S >= S_max) return V[N_s];

    const float idx_f = (S - S_min) / dS;
    int idx = static_cast<int>(floorf(idx_f));
    if (idx < 0) idx = 0;
    if (idx >= N_s) idx = N_s - 1;
    const float w = idx_f - idx;

    return (1.0f - w) * V[idx] + w * V[idx + 1];
}

__device__ float implied_vol_american_fd_cuda(
    const float target, const float S, const float K, const float T, const uint8_t is_call,
    const float* rates_curve,
    const float* rates_times,
    const int n_rates,
    const float pv_divs,
    const DivEvent* div_events,
    const int n_div_events,
    const float* r_sched,        // NEW
    const float* df_tT_sched,     // NEW
    const float tol         = 1e-6f,
    const int   max_iter    = 100,
    const float v_min       = 1e-4f,
    const float v_max       = 5.0f,
    const int   time_steps  = 200,
    const int   space_steps = 200)
{
    if (T <= 0.0f || S <= 0.0f || K <= 0.0f || target <= 0.0f)
        return 0.0f;

    // Check if price is at intrinsic
    float intrinsic = is_call ? fmaxf(S - K, 0.0f) : fmaxf(K - S, 0.0f);
    if (fabsf(target - intrinsic) < 1e-6f) {
        return v_min;
    }

    // Initial bracket check
    const float p_low = price_american_fd_cuda(
        S, K, T, v_min, is_call,
        rates_curve, rates_times, n_rates,
        pv_divs, div_events, n_div_events,
        time_steps, space_steps,
        r_sched, df_tT_sched);

    const float p_high = price_american_fd_cuda(
        S, K, T, v_max, is_call,
        rates_curve, rates_times, n_rates,
        pv_divs, div_events, n_div_events,
        time_steps, space_steps,
        r_sched, df_tT_sched);

    if (target <= p_low + 1e-9f) return v_min;
    if (target >= p_high - 1e-9f) return NAN;

    // Bisection with adaptive tolerance
    float lo = v_min;
    float hi = v_max;
    float mid = 0.0f;

    // Use relative tolerance for better convergence
    const float rel_tol = tol / fmaxf(target, 1.0f);

    for (int i = 0; i < max_iter; ++i) {
        mid = 0.5f * (lo + hi);

        const float p_mid = price_american_fd_cuda(
            S, K, T, mid, is_call,
            rates_curve, rates_times, n_rates,
            pv_divs, div_events, n_div_events,
            time_steps, space_steps,
            r_sched, df_tT_sched);

        const float diff = p_mid - target;
        const float rel_diff = fabsf(diff) / target;

        // Check convergence with both absolute and relative tolerance
        if (fabsf(diff) < tol || rel_diff < rel_tol) return mid;
        if (diff < 0.0f) lo = mid;
        else             hi = mid;

        // Check if bracket is tight enough
        if (hi - lo < v_min * 0.01f) break;
    }

    return mid;
}

__global__ void compute_fd_iv_kernel(
    const float *prices, const float *spots, const float *strikes, const float *tenors,
    const uint8_t *v_is_call, int n_options,
    const int *tenor_ids,
    const float *pv_divs_by_tenor,
    const DivEvent *div_events_by_tenor,
    const int *n_div_events_by_tenor,
    const float *r_by_tenor,           // NEW: size n_unique * RATE_SCHED_STRIDE
    const float *df_tT_by_tenor,       // NEW: size n_unique * RATE_SCHED_STRIDE
    const float *rates_curve, const float *rates_times, const int n_rates,
    float *results,
    const float tol, const int max_iter, const float v_min, const float v_max,
    const int time_steps, const int space_steps) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_options) return;

    const float p = prices[idx];
    const float s = spots[idx];
    const float k = strikes[idx];
    const float t = tenors[idx];
    const uint8_t is_call = v_is_call[idx];

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
    const float intrinsic = is_call ? fmaxf(s - k, 0.0f) : fmaxf(k - s, 0.0f);
    if (fabsf(p - intrinsic) < 1e-6f) {
        results[idx] = v_min;
        return;
    }

    // Check if price is reasonable (not below intrinsic)
    if (p < intrinsic - 1e-6f) {
        results[idx] = v_min;
        return;
    }

    int tid = tenor_ids[idx];
    float pv_divs = pv_divs_by_tenor[tid];
    const DivEvent* div_events = div_events_by_tenor + tid * MAX_DIV_EVENTS;
    int n_div_events = n_div_events_by_tenor[tid];

    const float* r_sched = (r_by_tenor != nullptr) ? (r_by_tenor + tid * RATE_SCHED_STRIDE) : nullptr;
    const float* df_sched = (df_tT_by_tenor != nullptr) ? (df_tT_by_tenor + tid * RATE_SCHED_STRIDE) : nullptr;


    results[idx] = implied_vol_american_fd_cuda(
        p, s, k, t, is_call,
        rates_curve, rates_times, n_rates,
        pv_divs, div_events, n_div_events,
        r_sched, df_sched,
        tol, max_iter, v_min, v_max,
        time_steps, space_steps);
}

__global__ void compute_fd_delta_kernel(
    const float* spots, const float* strikes, const float* tenors, const float* sigmas,
    const uint8_t* rights,
    const int* tenor_ids,
    const float* pv_divs_by_tenor,
    const DivEvent* div_events_by_tenor,
    const int* n_div_events_by_tenor,
    const float* r_by_tenor,
    const float* df_tT_by_tenor,
    const float* rates_curve, const float* rates_times, int n_rates,
    int time_steps, int space_steps,
    int n_options,
    float rel_shift,           // e.g. 1e-4
    float* out_delta)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_options) return;

    const float s = spots[idx];
    const float k = strikes[idx];
    const float t = tenors[idx];
    const float sigma = sigmas[idx];
    const uint8_t is_call = rights[idx];

    if (!isfinite(s) || !isfinite(k) || !isfinite(t) || !isfinite(sigma) ||
        t <= 0.0f || s <= 0.0f || k <= 0.0f || sigma <= 0.0f) {
        out_delta[idx] = NAN;
        return;
    }

    const int tid = tenor_ids[idx];
    const float pv_divs = pv_divs_by_tenor[tid];
    const DivEvent* div_events = div_events_by_tenor + tid * MAX_DIV_EVENTS;
    const int n_div_events = n_div_events_by_tenor[tid];

    const float* r_sched  = (r_by_tenor != nullptr)   ? (r_by_tenor + tid * RATE_SCHED_STRIDE) : nullptr;
    const float* df_sched = (df_tT_by_tenor != nullptr) ? (df_tT_by_tenor + tid * RATE_SCHED_STRIDE) : nullptr;

    float h = s * rel_shift;
    if (h < 1e-6f) h = 1e-6f;

    const float p_up = price_american_fd_cuda(
        s + h, k, t, sigma, is_call,
        rates_curve, rates_times, n_rates,
        pv_divs, div_events, n_div_events,
        time_steps, space_steps,
        r_sched, df_sched
    );

    const float p_dn = price_american_fd_cuda(
        fmaxf(s - h, 1e-8f), k, t, sigma, is_call,
        rates_curve, rates_times, n_rates,
        pv_divs, div_events, n_div_events,
        time_steps, space_steps,
        r_sched, df_sched
    );

    out_delta[idx] = (p_up - p_dn) / (2.0f * h);
}

__global__ void compute_fd_vega_kernel(
    const float* spots, const float* strikes, const float* tenors, const float* sigmas,
    const uint8_t* rights,
    const int* tenor_ids,
    const float* pv_divs_by_tenor,
    const DivEvent* div_events_by_tenor,
    const int* n_div_events_by_tenor,
    const float* r_by_tenor,
    const float* df_tT_by_tenor,
    const float* rates_curve, const float* rates_times, int n_rates,
    int time_steps, int space_steps,
    int n_options,
    float abs_shift,           // e.g. 1e-4
    float* out_vega)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_options) return;

    const float s = spots[idx];
    const float k = strikes[idx];
    const float t = tenors[idx];
    const float sigma = sigmas[idx];
    const uint8_t is_call = rights[idx];

    if (!isfinite(s) || !isfinite(k) || !isfinite(t) || !isfinite(sigma) ||
        t <= 0.0f || s <= 0.0f || k <= 0.0f || sigma <= 0.0f) {
        out_vega[idx] = NAN;
        return;
    }

    const int tid = tenor_ids[idx];
    const float pv_divs = pv_divs_by_tenor[tid];
    const DivEvent* div_events = div_events_by_tenor + tid * MAX_DIV_EVENTS;
    const int n_div_events = n_div_events_by_tenor[tid];

    const float* r_sched  = (r_by_tenor != nullptr)   ? (r_by_tenor + tid * RATE_SCHED_STRIDE) : nullptr;
    const float* df_sched = (df_tT_by_tenor != nullptr) ? (df_tT_by_tenor + tid * RATE_SCHED_STRIDE) : nullptr;

    float h = fmaxf(abs_shift, sigma * 1e-2f); // robust shift for large vols

    const float p_up = price_american_fd_cuda(
        s, k, t, sigma + h, is_call,
        rates_curve, rates_times, n_rates,
        pv_divs, div_events, n_div_events,
        time_steps, space_steps,
        r_sched, df_sched
    );

    const float sigma_dn = fmaxf(sigma - h, 1e-6f);
    const float p_dn = price_american_fd_cuda(
        s, k, t, sigma_dn, is_call,
        rates_curve, rates_times, n_rates,
        pv_divs, div_events, n_div_events,
        time_steps, space_steps,
        r_sched, df_sched
    );

    out_vega[idx] = (p_up - p_dn) / (2.0f * h);
}

// =======================
// Main IV kernel: no shared memory needed
// =======================

__global__ void compute_price_kernel_threadlocal(
    const float *spots, const float *strikes, const float *tenors, const float *sigmas,
    const uint8_t *rights,
    const int *tenor_ids,
    const float *pv_divs_by_tenor,
    const DivEvent *div_events_by_tenor,
    const int *n_div_events_by_tenor,
    const float *r_by_tenor,           // NEW: size n_unique * RATE_SCHED_STRIDE
    const float *df_tT_by_tenor,       // NEW: size n_unique * RATE_SCHED_STRIDE
    const float *rates_curve, const float *rates_times, int n_rates,
    const int time_steps,
    const int space_steps,
    const int n_options,
    float *out_price)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_options) return;

    const float s = spots[idx];
    const float k = strikes[idx];
    const float t = tenors[idx];
    const float sigma = sigmas[idx];
    const int right = rights[idx];

    if (!isfinite(s) || !isfinite(k) || !isfinite(t) || !isfinite(sigma) ||
        t <= 0.0f || s <= 0.0f || k <= 0.0f || sigma <= 0.0f) {
        if (t <= 0.0f || s <= 0.0f || k <= 0.0f || sigma <= 0.0f)
            out_price[idx] = right == 1 ? std::fmax(s - k, 0.0f) : std::fmax(k - s, 0.0f);
        return;
        }

    const int tid = tenor_ids[idx];
    const float pv_divs = pv_divs_by_tenor[tid];
    const DivEvent* div_events = div_events_by_tenor + tid * MAX_DIV_EVENTS;
    const int n_div_events = n_div_events_by_tenor[tid];

    const float* r_sched = (r_by_tenor != nullptr) ? (r_by_tenor + tid * RATE_SCHED_STRIDE) : nullptr;
    const float* df_sched = (df_tT_by_tenor != nullptr) ? (df_tT_by_tenor + tid * RATE_SCHED_STRIDE) : nullptr;

    out_price[idx] = price_american_fd_cuda(
        s, k, t, sigma, right,
        rates_curve, rates_times, n_rates,
        pv_divs, div_events, n_div_events,
        time_steps, space_steps,
        r_sched, df_sched
    );
}


/////////////////////////////////////////////////////////
/// Host / CPU wrappers that launch kernels
/////////////////////////////////////////////////////////

__global__ void build_rates_schedule_kernel(
    const float* unique_tenors, int n_unique,
    int time_steps,
    const float* rates_curve, const float* rates_times, int n_rates,
    float* out_r_by_tenor,     // size n_unique * RATE_SCHED_STRIDE
    float* out_df_tT_by_tenor) // size n_unique * RATE_SCHED_STRIDE
{
    const int tid = blockIdx.x;
    const int n   = threadIdx.x;

    if (tid >= n_unique) return;
    if (n >= RATE_SCHED_STRIDE) return;

    const float T = unique_tenors[tid];

    // Mirror FD clamping behavior
    int N_t = time_steps;
    if (N_t < 10) N_t = 10;
    if (N_t > MAX_TIME_STEPS_SCHED) N_t = MAX_TIME_STEPS_SCHED;

    const float dt = (T > 0.0f) ? (T / static_cast<float>(N_t)) : 0.0f;

    // Only fill valid indices; pad the rest with terminal values so accidental reads are benign.
    const int max_n = N_t; // inclusive, we store 0..N_t
    float r = 0.0f;
    float df = 1.0f;

    if (n <= max_n && T > 0.0f && n_rates > 0) {
        const float t = static_cast<float>(n) * dt;
        r  = get_rate_at_time_convex_monotone(t, rates_curve, rates_times, n_rates);
        df = df_t_T(t, T, rates_curve, rates_times, n_rates);
    } else if (n > max_n && T > 0.0f && n_rates > 0) {
        // Pad with values at maturity t=T
        r  = get_rate_at_time_convex_monotone(T, rates_curve, rates_times, n_rates);
        df = 1.0f; // D(T,T)
    } else {
        r  = 0.0f;
        df = 1.0f;
    }

    out_r_by_tenor[tid * RATE_SCHED_STRIDE + n] = r;
    out_df_tT_by_tenor[tid * RATE_SCHED_STRIDE + n] = df;
}


static float get_rate_at_time_host(
    const float t, const std::vector<float>& rates_curve, const std::vector<float>& rates_times)
{
    const int n_points = static_cast<int>(rates_curve.size());
    if (n_points <= 0) return 0.0f;
    if (t <= rates_times[0]) return rates_curve[0];
    if (t >= rates_times[n_points - 1]) return rates_curve[n_points - 1];

    for (int i = 0; i < n_points - 1; ++i) {
        if (t <= rates_times[i + 1]) {
            const float t_low = rates_times[i];
            const float t_high = rates_times[i + 1];
            const float r_low = rates_curve[i];
            const float r_high = rates_curve[i + 1];
            return (r_high * t_high - r_low * t_low) / (t_high - t_low);
        }
    }
    return rates_curve[n_points - 1];
}

static void precompute_dividends_host(
    const float T, const int time_steps,
    const std::vector<float>& rates_curve, const std::vector<float>& rates_times,
    const std::vector<float>& div_amounts, const std::vector<float>& div_times,
    float& out_pv_divs,
    DivEvent* out_div_events,
    int& out_n_div_events,
    const float div_scale = 1.0f)
{
    out_pv_divs = 0.0f;
    out_n_div_events = 0;

    const int n_divs = static_cast<int>(div_amounts.size());

    for (int i = 0; i < n_divs; ++i) {
        const float t_div = div_times[i];
        if (t_div > 0.0f && t_div <= T) {
            const float r_div = get_rate_at_time_host(t_div, rates_curve, rates_times);
            out_pv_divs += div_amounts[i] * div_scale * std::exp(-r_div * t_div);
        }
    }

    float dt = T / static_cast<float>(time_steps);
    for (int k = 0; k < n_divs && out_n_div_events < MAX_DIV_EVENTS; ++k) {
        float tD = div_times[k];
        if (tD > 0.0f && tD < T) {
            int step = static_cast<int>(std::floor(tD / dt));
            if (step < 0) step = 0;
            if (step >= time_steps) step = time_steps - 1;
            out_div_events[out_n_div_events++] = {step, div_amounts[k] * div_scale, tD};
        }
    }
}

struct TenorDividendScheduleHost {
    std::vector<int> tenor_ids;                 // size n_options
    std::vector<float> unique_tenors;           // size n_unique
    std::vector<float> pv_divs_by_tenor;        // size n_unique
    std::vector<int> n_div_events_by_tenor;     // size n_unique
    std::vector<DivEvent> div_events_by_tenor;  // size n_unique * MAX_DIV_EVENTS
};

static TenorDividendScheduleHost build_tenor_dividend_schedule_host(
    const std::vector<float>& tenors,
    const int time_steps,
    const std::vector<float>& rates_curve,
    const std::vector<float>& rates_times,
    const std::vector<float>& div_amounts,
    const std::vector<float>& div_times,
    const float S_ref = 1.0f)
{
    TenorDividendScheduleHost out;

    const int n_options = static_cast<int>(tenors.size());

    // Bucket tenors by day to avoid exploding unique tenor count after switching to datetime-based tenors.
    // We map each tenor T (in years) to an integer day bucket: day = round(T * 365).
    // Then we store the representative unique tenor as day / 365.
    constexpr float DAYS_PER_YEAR = 365.0f;

    std::unordered_map<int, int> day_to_id;
    day_to_id.reserve(64);

    out.unique_tenors.reserve(64);
    out.tenor_ids.assign(n_options, 0);

    for (int i = 0; i < n_options; ++i) {
        const float T = tenors[i];

        // Handle non-finite / negative tenors deterministically (they shouldn't appear in pricing anyway).
        const int day_bucket = (std::isfinite(T) && T > 0.0f)
            ? static_cast<int>(std::lround(T * DAYS_PER_YEAR))
            : 0;

        auto it = day_to_id.find(day_bucket);
        if (it == day_to_id.end()) {
            const int new_id = static_cast<int>(out.unique_tenors.size());
            const float T_bucket = static_cast<float>(day_bucket) / DAYS_PER_YEAR;

            out.unique_tenors.push_back(T_bucket);
            day_to_id.emplace(day_bucket, new_id);
            out.tenor_ids[i] = new_id;
        } else {
            out.tenor_ids[i] = it->second;
        }
    }

    const int n_unique = static_cast<int>(out.unique_tenors.size());
    out.pv_divs_by_tenor.assign(n_unique, 0.0f);
    out.n_div_events_by_tenor.assign(n_unique, 0);
    out.div_events_by_tenor.resize(static_cast<size_t>(n_unique) * MAX_DIV_EVENTS);

    const float div_scale = (std::isfinite(S_ref) && S_ref > 0.0f) ? (1.0f / S_ref) : 1.0f;

    for (int tid = 0; tid < n_unique; ++tid) {
        float pv = 0.0f;
        int n_ev = 0;
        DivEvent* ev = out.div_events_by_tenor.data() + tid * MAX_DIV_EVENTS;

        precompute_dividends_host(
            out.unique_tenors[tid], time_steps,
            rates_curve, rates_times,
            div_amounts, div_times,
            pv, ev, n_ev,
            div_scale
        );

        out.pv_divs_by_tenor[tid] = pv;
        out.n_div_events_by_tenor[tid] = n_ev;
    }

    return out;
}


std::vector<float> get_v_iv_fd_cuda(
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
    const int n_options = prices.size();
    std::vector results(n_options, std::numeric_limits<float>::quiet_NaN());

    if (n_options == 0) return results;

    // Validate input sizes
    if (spots.size() != n_options || strikes.size() != n_options ||
        tenors.size() != n_options || v_is_call.size() != n_options) {
        return results;
    }

    const int n_rates = rates_curve.size();

    // Precompute per-tenor dividend schedules on host (only ~20 tenors expected)
    const TenorDividendScheduleHost sched = build_tenor_dividend_schedule_host(
        tenors, time_steps, rates_curve, rates_times, div_amounts, div_times
    );
    const int n_unique = static_cast<int>(sched.unique_tenors.size());

    // Validate rate and dividend data
    if (!rates_curve.empty() && rates_curve.size() != rates_times.size()) {
        return results;
    }
    if (!div_amounts.empty() && div_amounts.size() != div_times.size()) {
        return results;
    }

    float *d_prices, *d_spots, *d_strikes, *d_tenors, *d_results;
    uint8_t *d_v_is_call;

    int *d_tenor_ids = nullptr;
    float *d_pv_divs_by_tenor = nullptr;
    DivEvent *d_div_events_by_tenor = nullptr;
    int *d_n_div_events_by_tenor = nullptr;

    float *d_rates_curve = nullptr, *d_rates_times = nullptr;

    // NEW: per-tenor rate schedules
    float *d_unique_tenors = nullptr;
    float *d_r_by_tenor = nullptr;
    float *d_df_tT_by_tenor = nullptr;

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

    cudaMalloc(&d_tenor_ids, n_options * sizeof(int));
    cudaMemcpy(d_tenor_ids, sched.tenor_ids.data(), n_options * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_pv_divs_by_tenor, n_unique * sizeof(float));
    cudaMalloc(&d_n_div_events_by_tenor, n_unique * sizeof(int));
    cudaMalloc(&d_div_events_by_tenor, static_cast<size_t>(n_unique) * MAX_DIV_EVENTS * sizeof(DivEvent));

    cudaMemcpy(d_pv_divs_by_tenor, sched.pv_divs_by_tenor.data(), n_unique * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_n_div_events_by_tenor, sched.n_div_events_by_tenor.data(), n_unique * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_div_events_by_tenor, sched.div_events_by_tenor.data(),
               static_cast<size_t>(n_unique) * MAX_DIV_EVENTS * sizeof(DivEvent), cudaMemcpyHostToDevice);

    if (n_rates > 0) {
        cudaMalloc(&d_rates_curve, n_rates * sizeof(float));
        cudaMalloc(&d_rates_times, n_rates * sizeof(float));
        cudaMemcpy(d_rates_curve, rates_curve.data(), n_rates * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_rates_times, rates_times.data(), n_rates * sizeof(float), cudaMemcpyHostToDevice);
    }

    // NEW: build per-tenor rate/discount schedules on device
    if (n_unique > 0) {
        cudaMalloc(&d_unique_tenors, n_unique * sizeof(float));
        cudaMemcpy(d_unique_tenors, sched.unique_tenors.data(), n_unique * sizeof(float), cudaMemcpyHostToDevice);

        cudaMalloc(&d_r_by_tenor, static_cast<size_t>(n_unique) * RATE_SCHED_STRIDE * sizeof(float));
        cudaMalloc(&d_df_tT_by_tenor, static_cast<size_t>(n_unique) * RATE_SCHED_STRIDE * sizeof(float));

        // One block per tenor, 257 threads to fill [0..256]
        build_rates_schedule_kernel<<<n_unique, RATE_SCHED_STRIDE>>>(
            d_unique_tenors, n_unique,
            time_steps,
            d_rates_curve, d_rates_times, n_rates,
            d_r_by_tenor,
            d_df_tT_by_tenor
        );
    }

    int block_size = 256;
    int grid_size = (n_options + block_size - 1) / block_size;

    compute_fd_iv_kernel<<<grid_size, block_size>>>(
        d_prices, d_spots, d_strikes, d_tenors, d_v_is_call, n_options,
        d_tenor_ids,
        d_pv_divs_by_tenor,
        d_div_events_by_tenor,
        d_n_div_events_by_tenor,
        d_r_by_tenor,
        d_df_tT_by_tenor,
        d_rates_curve, d_rates_times, n_rates,
        d_results,
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

    cudaFree(d_tenor_ids);
    cudaFree(d_pv_divs_by_tenor);
    cudaFree(d_div_events_by_tenor);
    cudaFree(d_n_div_events_by_tenor);

    // NEW: free schedules
    if (d_unique_tenors) cudaFree(d_unique_tenors);
    if (d_r_by_tenor) cudaFree(d_r_by_tenor);
    if (d_df_tT_by_tenor) cudaFree(d_df_tT_by_tenor);

    if (d_rates_curve) cudaFree(d_rates_curve);
    if (d_rates_times) cudaFree(d_rates_times);

    return results;
}

std::vector<float> get_v_price_fd_cuda(
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
    const int n_options = spots.size();
    std::vector<float> prices(n_options);

    if (n_options == 0) return prices;

    const int n_rates = rates_curve.size();

    // Precompute per-tenor dividend schedules on host (only ~20 tenors expected)
    TenorDividendScheduleHost sched = build_tenor_dividend_schedule_host(
        tenors, time_steps, rates_curve, rates_times, div_amounts, div_times
    );
    const int n_unique = static_cast<int>(sched.unique_tenors.size());

    float *d_spots, *d_strikes, *d_tenors, *d_sigmas;
    uint8_t *d_v_is_call;
    float *d_prices;

    int *d_tenor_ids = nullptr;
    float *d_pv_divs_by_tenor = nullptr;
    DivEvent *d_div_events_by_tenor = nullptr;
    int *d_n_div_events_by_tenor = nullptr;

    float *d_rates_curve = nullptr, *d_rates_times = nullptr;

    // NEW: per-tenor rate schedules
    float *d_unique_tenors = nullptr;
    float *d_r_by_tenor = nullptr;
    float *d_df_tT_by_tenor = nullptr;

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

    cudaMalloc(&d_tenor_ids, n_options * sizeof(int));
    cudaMemcpy(d_tenor_ids, sched.tenor_ids.data(), n_options * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_pv_divs_by_tenor, n_unique * sizeof(float));
    cudaMalloc(&d_n_div_events_by_tenor, n_unique * sizeof(int));
    cudaMalloc(&d_div_events_by_tenor, static_cast<size_t>(n_unique) * MAX_DIV_EVENTS * sizeof(DivEvent));

    cudaMemcpy(d_pv_divs_by_tenor, sched.pv_divs_by_tenor.data(), n_unique * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_n_div_events_by_tenor, sched.n_div_events_by_tenor.data(), n_unique * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_div_events_by_tenor, sched.div_events_by_tenor.data(),
               static_cast<size_t>(n_unique) * MAX_DIV_EVENTS * sizeof(DivEvent), cudaMemcpyHostToDevice);

    if (n_rates > 0 && rates_curve.size() == rates_times.size()) {
        cudaMalloc(&d_rates_curve, n_rates * sizeof(float));
        cudaMalloc(&d_rates_times, n_rates * sizeof(float));
        cudaMemcpy(d_rates_curve, rates_curve.data(), n_rates * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_rates_times, rates_times.data(), n_rates * sizeof(float), cudaMemcpyHostToDevice);
    }

    // NEW: build per-tenor rate/discount schedules on device
    if (n_unique > 0) {
        cudaMalloc(&d_unique_tenors, n_unique * sizeof(float));
        cudaMemcpy(d_unique_tenors, sched.unique_tenors.data(), n_unique * sizeof(float), cudaMemcpyHostToDevice);

        cudaMalloc(&d_r_by_tenor, static_cast<size_t>(n_unique) * RATE_SCHED_STRIDE * sizeof(float));
        cudaMalloc(&d_df_tT_by_tenor, static_cast<size_t>(n_unique) * RATE_SCHED_STRIDE * sizeof(float));

        // One block per tenor, 257 threads to fill [0..256]
        build_rates_schedule_kernel<<<n_unique, RATE_SCHED_STRIDE>>>(
            d_unique_tenors, n_unique,
            time_steps,
            d_rates_curve, d_rates_times, n_rates,
            d_r_by_tenor,
            d_df_tT_by_tenor
        );
    }

    int block_size = 256;
    int grid_size = (n_options + block_size - 1) / block_size;

    compute_price_kernel_threadlocal<<<grid_size, block_size>>>(
        d_spots, d_strikes, d_tenors, d_sigmas, d_v_is_call,
        d_tenor_ids,
        d_pv_divs_by_tenor,
        d_div_events_by_tenor,
        d_n_div_events_by_tenor,
        d_r_by_tenor,
        d_df_tT_by_tenor,
        d_rates_curve, d_rates_times, n_rates,
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

    cudaFree(d_tenor_ids);
    cudaFree(d_pv_divs_by_tenor);
    cudaFree(d_div_events_by_tenor);
    cudaFree(d_n_div_events_by_tenor);

    // NEW: free schedules
    if (d_unique_tenors) cudaFree(d_unique_tenors);
    if (d_r_by_tenor) cudaFree(d_r_by_tenor);
    if (d_df_tT_by_tenor) cudaFree(d_df_tT_by_tenor);

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
    const std::vector<float> &div_times)
{
    const int n_options = static_cast<int>(spots.size());
    std::vector<float> deltas(n_options, std::numeric_limits<float>::quiet_NaN());
    if (n_options == 0) return deltas;

    if (strikes.size() != static_cast<size_t>(n_options) ||
        tenors.size()  != static_cast<size_t>(n_options) ||
        v_is_call.size()!= static_cast<size_t>(n_options) ||
        sigmas.size()  != static_cast<size_t>(n_options)) {
        return deltas;
    }

    if (!rates_curve.empty() && rates_curve.size() != rates_times.size()) return deltas;
    if (!div_amounts.empty() && div_amounts.size() != div_times.size()) return deltas;

    // Per-tenor dividend schedules (and tenors bucketing) like price/IV
    const int time_steps = 200;   // FD grid time steps (keep consistent with your pricer defaults)
    const int space_steps = 200;  // FD grid space steps
    const TenorDividendScheduleHost sched = build_tenor_dividend_schedule_host(
        tenors, time_steps, rates_curve, rates_times, div_amounts, div_times
    );
    const int n_unique = static_cast<int>(sched.unique_tenors.size());
    const int n_rates = static_cast<int>(rates_curve.size());

    float *d_spots=nullptr, *d_strikes=nullptr, *d_tenors=nullptr, *d_sigmas=nullptr, *d_out=nullptr;
    uint8_t *d_rights=nullptr;

    int *d_tenor_ids=nullptr;
    float *d_pv_divs_by_tenor=nullptr;
    DivEvent *d_div_events_by_tenor=nullptr;
    int *d_n_div_events_by_tenor=nullptr;

    float *d_rates_curve=nullptr, *d_rates_times=nullptr;

    float *d_unique_tenors=nullptr, *d_r_by_tenor=nullptr, *d_df_tT_by_tenor=nullptr;

    cudaMalloc(&d_spots,   n_options * sizeof(float));
    cudaMalloc(&d_strikes, n_options * sizeof(float));
    cudaMalloc(&d_tenors,  n_options * sizeof(float));
    cudaMalloc(&d_sigmas,  n_options * sizeof(float));
    cudaMalloc(&d_rights,  n_options * sizeof(uint8_t));
    cudaMalloc(&d_out,     n_options * sizeof(float));

    cudaMemcpy(d_spots,   spots.data(),   n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_strikes, strikes.data(), n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tenors,  tenors.data(),  n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sigmas,  sigmas.data(),  n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rights,  v_is_call.data(), n_options * sizeof(uint8_t), cudaMemcpyHostToDevice);

    cudaMalloc(&d_tenor_ids, n_options * sizeof(int));
    cudaMemcpy(d_tenor_ids, sched.tenor_ids.data(), n_options * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_pv_divs_by_tenor, n_unique * sizeof(float));
    cudaMalloc(&d_n_div_events_by_tenor, n_unique * sizeof(int));
    cudaMalloc(&d_div_events_by_tenor, static_cast<size_t>(n_unique) * MAX_DIV_EVENTS * sizeof(DivEvent));

    cudaMemcpy(d_pv_divs_by_tenor, sched.pv_divs_by_tenor.data(), n_unique * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_n_div_events_by_tenor, sched.n_div_events_by_tenor.data(), n_unique * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_div_events_by_tenor, sched.div_events_by_tenor.data(),
               static_cast<size_t>(n_unique) * MAX_DIV_EVENTS * sizeof(DivEvent), cudaMemcpyHostToDevice);

    if (n_rates > 0) {
        cudaMalloc(&d_rates_curve, n_rates * sizeof(float));
        cudaMalloc(&d_rates_times, n_rates * sizeof(float));
        cudaMemcpy(d_rates_curve, rates_curve.data(), n_rates * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_rates_times, rates_times.data(), n_rates * sizeof(float), cudaMemcpyHostToDevice);
    }

    // Precompute per-tenor rate/df schedules on GPU
    if (n_unique > 0) {
        cudaMalloc(&d_unique_tenors, n_unique * sizeof(float));
        cudaMemcpy(d_unique_tenors, sched.unique_tenors.data(), n_unique * sizeof(float), cudaMemcpyHostToDevice);

        cudaMalloc(&d_r_by_tenor, static_cast<size_t>(n_unique) * RATE_SCHED_STRIDE * sizeof(float));
        cudaMalloc(&d_df_tT_by_tenor, static_cast<size_t>(n_unique) * RATE_SCHED_STRIDE * sizeof(float));

        build_rates_schedule_kernel<<<n_unique, RATE_SCHED_STRIDE>>>(
            d_unique_tenors, n_unique,
            time_steps,
            d_rates_curve, d_rates_times, n_rates,
            d_r_by_tenor,
            d_df_tT_by_tenor
        );
    }

    const int block_size = 256;
    const int grid_size = (n_options + block_size - 1) / block_size;

    // rel_shift: use something small; n_steps is kept for API compatibility (FD delta uses shift, not binomial steps)
    const float rel_shift = 1e-4f;
    compute_fd_delta_kernel<<<grid_size, block_size>>>(
        d_spots, d_strikes, d_tenors, d_sigmas, d_rights,
        d_tenor_ids,
        d_pv_divs_by_tenor,
        d_div_events_by_tenor,
        d_n_div_events_by_tenor,
        d_r_by_tenor,
        d_df_tT_by_tenor,
        d_rates_curve, d_rates_times, n_rates,
        time_steps, space_steps,
        n_options,
        rel_shift,
        d_out
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }

    cudaMemcpy(deltas.data(), d_out, n_options * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_spots);
    cudaFree(d_strikes);
    cudaFree(d_tenors);
    cudaFree(d_sigmas);
    cudaFree(d_rights);
    cudaFree(d_out);

    cudaFree(d_tenor_ids);
    cudaFree(d_pv_divs_by_tenor);
    cudaFree(d_div_events_by_tenor);
    cudaFree(d_n_div_events_by_tenor);

    if (d_unique_tenors) cudaFree(d_unique_tenors);
    if (d_r_by_tenor) cudaFree(d_r_by_tenor);
    if (d_df_tT_by_tenor) cudaFree(d_df_tT_by_tenor);

    if (d_rates_curve) cudaFree(d_rates_curve);
    if (d_rates_times) cudaFree(d_rates_times);

    (void)n_steps; // API compatibility: currently not used in FD delta
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
    const std::vector<float> &div_times)
{
    const int n_options = static_cast<int>(spots.size());
    std::vector<float> vegas(n_options, std::numeric_limits<float>::quiet_NaN());
    if (n_options == 0) return vegas;

    if (strikes.size() != static_cast<size_t>(n_options) ||
        tenors.size()  != static_cast<size_t>(n_options) ||
        v_is_call.size()!= static_cast<size_t>(n_options) ||
        sigmas.size()  != static_cast<size_t>(n_options)) {
        return vegas;
    }

    if (!rates_curve.empty() && rates_curve.size() != rates_times.size()) return vegas;
    if (!div_amounts.empty() && div_amounts.size() != div_times.size()) return vegas;

    const int time_steps = 200;
    const int space_steps = 200;
    const TenorDividendScheduleHost sched = build_tenor_dividend_schedule_host(
        tenors, time_steps, rates_curve, rates_times, div_amounts, div_times
    );
    const int n_unique = static_cast<int>(sched.unique_tenors.size());
    const int n_rates = static_cast<int>(rates_curve.size());

    float *d_spots=nullptr, *d_strikes=nullptr, *d_tenors=nullptr, *d_sigmas=nullptr, *d_out=nullptr;
    uint8_t *d_rights=nullptr;

    int *d_tenor_ids=nullptr;
    float *d_pv_divs_by_tenor=nullptr;
    DivEvent *d_div_events_by_tenor=nullptr;
    int *d_n_div_events_by_tenor=nullptr;

    float *d_rates_curve=nullptr, *d_rates_times=nullptr;

    float *d_unique_tenors=nullptr, *d_r_by_tenor=nullptr, *d_df_tT_by_tenor=nullptr;

    cudaMalloc(&d_spots,   n_options * sizeof(float));
    cudaMalloc(&d_strikes, n_options * sizeof(float));
    cudaMalloc(&d_tenors,  n_options * sizeof(float));
    cudaMalloc(&d_sigmas,  n_options * sizeof(float));
    cudaMalloc(&d_rights,  n_options * sizeof(uint8_t));
    cudaMalloc(&d_out,     n_options * sizeof(float));

    cudaMemcpy(d_spots,   spots.data(),   n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_strikes, strikes.data(), n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tenors,  tenors.data(),  n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sigmas,  sigmas.data(),  n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rights,  v_is_call.data(), n_options * sizeof(uint8_t), cudaMemcpyHostToDevice);

    cudaMalloc(&d_tenor_ids, n_options * sizeof(int));
    cudaMemcpy(d_tenor_ids, sched.tenor_ids.data(), n_options * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_pv_divs_by_tenor, n_unique * sizeof(float));
    cudaMalloc(&d_n_div_events_by_tenor, n_unique * sizeof(int));
    cudaMalloc(&d_div_events_by_tenor, static_cast<size_t>(n_unique) * MAX_DIV_EVENTS * sizeof(DivEvent));

    cudaMemcpy(d_pv_divs_by_tenor, sched.pv_divs_by_tenor.data(), n_unique * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_n_div_events_by_tenor, sched.n_div_events_by_tenor.data(), n_unique * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_div_events_by_tenor, sched.div_events_by_tenor.data(),
               static_cast<size_t>(n_unique) * MAX_DIV_EVENTS * sizeof(DivEvent), cudaMemcpyHostToDevice);

    if (n_rates > 0) {
        cudaMalloc(&d_rates_curve, n_rates * sizeof(float));
        cudaMalloc(&d_rates_times, n_rates * sizeof(float));
        cudaMemcpy(d_rates_curve, rates_curve.data(), n_rates * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_rates_times, rates_times.data(), n_rates * sizeof(float), cudaMemcpyHostToDevice);
    }

    if (n_unique > 0) {
        cudaMalloc(&d_unique_tenors, n_unique * sizeof(float));
        cudaMemcpy(d_unique_tenors, sched.unique_tenors.data(), n_unique * sizeof(float), cudaMemcpyHostToDevice);

        cudaMalloc(&d_r_by_tenor, static_cast<size_t>(n_unique) * RATE_SCHED_STRIDE * sizeof(float));
        cudaMalloc(&d_df_tT_by_tenor, static_cast<size_t>(n_unique) * RATE_SCHED_STRIDE * sizeof(float));

        build_rates_schedule_kernel<<<n_unique, RATE_SCHED_STRIDE>>>(
            d_unique_tenors, n_unique,
            time_steps,
            d_rates_curve, d_rates_times, n_rates,
            d_r_by_tenor,
            d_df_tT_by_tenor
        );
    }

    const int block_size = 256;
    const int grid_size = (n_options + block_size - 1) / block_size;

    const float abs_shift = 1e-4f;
    compute_fd_vega_kernel<<<grid_size, block_size>>>(
        d_spots, d_strikes, d_tenors, d_sigmas, d_rights,
        d_tenor_ids,
        d_pv_divs_by_tenor,
        d_div_events_by_tenor,
        d_n_div_events_by_tenor,
        d_r_by_tenor,
        d_df_tT_by_tenor,
        d_rates_curve, d_rates_times, n_rates,
        time_steps, space_steps,
        n_options,
        abs_shift,
        d_out
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }

    cudaMemcpy(vegas.data(), d_out, n_options * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_spots);
    cudaFree(d_strikes);
    cudaFree(d_tenors);
    cudaFree(d_sigmas);
    cudaFree(d_rights);
    cudaFree(d_out);

    cudaFree(d_tenor_ids);
    cudaFree(d_pv_divs_by_tenor);
    cudaFree(d_div_events_by_tenor);
    cudaFree(d_n_div_events_by_tenor);

    if (d_unique_tenors) cudaFree(d_unique_tenors);
    if (d_r_by_tenor) cudaFree(d_r_by_tenor);
    if (d_df_tT_by_tenor) cudaFree(d_df_tT_by_tenor);

    if (d_rates_curve) cudaFree(d_rates_curve);
    if (d_rates_times) cudaFree(d_rates_times);

    (void)n_steps; // API compatibility: currently not used in FD vega
    return vegas;
}