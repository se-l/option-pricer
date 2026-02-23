#include <algorithm>
#include <cmath>
#include <vector>
#include <stdexcept>


static float get_rate_at_time_host(float t, const float *rates_curve,
                                   const float *rates_times, int n_points) {
    if (n_points <= 0) return 0.0f;
    if (t <= rates_times[0]) return rates_curve[0];
    if (t >= rates_times[n_points - 1]) {
        // Beyond the last point, assume last instantaneous forward continues
        return rates_curve[n_points - 1];
    }

    for (int i = 0; i < n_points - 1; ++i) {
        if (t <= rates_times[i + 1]) {
            float t_low  = rates_times[i];
            float t_high = rates_times[i + 1];
            float r_low  = rates_curve[i];
            float r_high = rates_curve[i + 1];
            return (r_high * t_high - r_low * t_low) / (t_high - t_low);
        }
    }
    return rates_curve[n_points - 1];
}

static float get_rate_at_time_convex_monotone_host(
    float t,
    const float* rates_curve,
    const float* rates_times,
    int n_points)
{
    if (n_points <= 0) return 0.0f;
    if (t <= 0.0f) {
        // Best-effort: use first available rate as short rate proxy.
        return rates_curve[0];
    }

    // Clamp outside curve domain
    if (t <= rates_times[0]) return rates_curve[0];
    if (t >= rates_times[n_points - 1]) return rates_curve[n_points - 1];

    // Find interval k such that t in (t_k, t_{k+1}]
    int k = 0;
    for (int i = 0; i < n_points - 1; ++i) {
        if (t <= rates_times[i + 1]) { k = i; break; }
    }

    // Helper lambdas (kept local to avoid polluting file scope)
    auto Z = [&](int i) -> float { return rates_curve[i] * rates_times[i]; };
    auto f_interval = [&](int i) -> float {
        const float t0 = rates_times[i];
        const float t1 = rates_times[i + 1];
        const float h = t1 - t0;
        if (!(h > 0.0f)) return 0.0f;
        return (Z(i + 1) - Z(i)) / h;
    };
    auto clampf = [&](float x, float lo, float hi) -> float {
        return fminf(fmaxf(x, lo), hi);
    };

    // Node forwards fhat_i with monotone limiter (Fritsch-Carlson style on slopes)
    auto fhat = [&](int i) -> float {
        if (i <= 0) return f_interval(0);
        if (i >= n_points - 1) return f_interval(n_points - 2);

        const float fL = f_interval(i - 1);
        const float fR = f_interval(i);

        // If slope changes sign, set to 0 to avoid overshoot
        if (fL * fR <= 0.0f) return 0.0f;

        // Weighted harmonic mean for stability and monotonicity
        const float hL = rates_times[i] - rates_times[i - 1];
        const float hR = rates_times[i + 1] - rates_times[i];
        const float wL = 2.0f * hR + hL;
        const float wR = hR + 2.0f * hL;
        const float denom = (wL / fL) + (wR / fR);
        if (fabsf(denom) < 1e-20f) return 0.0f;
        float fh = (wL + wR) / denom;

        // Additional bounding: keep within neighboring interval forwards
        const float lo = fminf(fL, fR);
        const float hi = fmaxf(fL, fR);
        return clampf(fh, lo, hi);
    };

    const float t0 = rates_times[k];
    const float t1 = rates_times[k + 1];
    const float h = t1 - t0;
    if (!(h > 0.0f)) return rates_curve[k];

    const float x = (t - t0) / h; // in (0,1]
    const float fbar = f_interval(k);     // average forward over interval
    const float fL = fhat(k);             // node forward at left
    const float fR = fhat(k + 1);         // node forward at right

    // Quadratic forward: f(x) = fL + (fR-fL)*x + c*x*(1-x)
    // Enforce average over [0,1] equals fbar => c = 6*(fbar - (fL+fR)/2)
    float c = 6.0f * (fbar - 0.5f * (fL + fR));

    // Monotone safeguard: ensure midpoint doesn't overshoot endpoint range
    const float f_mid = 0.5f * (fL + fR) + 0.25f * c;
    const float lo = fminf(fL, fR);
    const float hi = fmaxf(fL, fR);
    if (f_mid < lo || f_mid > hi) {
        const float target = (f_mid < lo) ? lo : hi;
        // f_mid = (fL+fR)/2 + c/4  =>  c = 4*(target - (fL+fR)/2)
        c = 4.0f * (target - 0.5f * (fL + fR));
    }

    float f = fL + (fR - fL) * x + c * x * (1.0f - x);

    // Final clamp (very conservative; avoids NaN/Inf propagation)
    f = clampf(f, lo - 1e6f, hi + 1e6f);
    if (!std::isfinite(f)) return rates_curve[k];
    return f;
}


// Simple tridiagonal solver (Thomas algorithm) for A * x = d
static void solve_tridiagonal(
    int n,
    const std::vector<float>& a,  // sub-diagonal [1..n-1]
    const std::vector<float>& b,  // diagonal    [0..n-1]
    const std::vector<float>& c,  // super-diag  [0..n-2]
    std::vector<float>&       d)  // RHS         [0..n-1], overwritten by solution
{
    std::vector<float> c_star(n);
    std::vector<float> d_star(n);

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

static float get_Z_at_time_host(float t, const float* rates_curve,
                               const float* rates_times, int n_points) {
    // Z(t) = R(t) * t with linear interpolation on the (time, Z) points.
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

static float df_0_t_host(float t, const float* rates_curve,
                         const float* rates_times, int n_points) {
    return std::expf(-get_Z_at_time_host(t, rates_curve, rates_times, n_points));
}

static float df_t_T_host(float t, float T, const float* rates_curve,
                         const float* rates_times, int n_points) {
    float Zt = get_Z_at_time_host(t, rates_curve, rates_times, n_points);
    float ZT = get_Z_at_time_host(T, rates_curve, rates_times, n_points);
    return std::expf(-(ZT - Zt));
}

float price_american_fd_div_host(
    float S, float K, float T,
    float sigma, uint8_t is_call,
    const std::vector<float>& rates_curve,
    const std::vector<float>& rates_times,
    const std::vector<float>& div_amounts,
    const std::vector<float>& div_times,
    int time_steps = 200,
    int space_steps = 200)
{
    if (T <= 0.0f || S <= 0.0f || K <= 0.0f || sigma <= 0.0f)
        return is_call ? std::fmax(S - K, 0.0f) : std::fmax(K - S, 0.0f);

    const int N_t = (time_steps < 10) ? 10 : time_steps;
    const int N_s = (space_steps < 10) ? 10 : space_steps;

    const float dt = T / static_cast<float>(N_t);

    const float* rc = rates_curve.empty() ? nullptr : rates_curve.data();
    const float* rt = rates_times.empty() ? nullptr : rates_times.data();
    const int n_rates = static_cast<int>(rates_curve.size());

    // Calculate PV of dividends for grid construction
    float pv_divs = 0.0f;
    for (std::size_t i = 0; i < div_amounts.size() && i < div_times.size(); ++i) {
        float t_div = div_times[i];
        if (t_div > 0.0f && t_div <= T) {
            pv_divs += div_amounts[i] * df_0_t_host(t_div, rc, rt, n_rates);
        }
    }

    // float S_grid_center = std::fmax(S - pv_divs, S * 0.1f);

    // Build spatial grid - ensure it covers likely dividend scenarios
    float r0 = get_rate_at_time_convex_monotone_host(0.0f,
                                     rates_curve.empty() ? nullptr : rates_curve.data(),
                                     rates_times.empty() ? nullptr : rates_times.data(),
                                     static_cast<int>(rates_curve.size()));

    // For puts, ensure grid goes low enough to capture deep ITM scenarios
    float S_min = 0.0f;  // Always start at zero for puts
    float S_max = std::fmaxf(
        (S + pv_divs) *
        std::expf(fminf(r0 * T + 4.0f * sigma * sqrt(T), 80.0f)), // 80 protects against overflow
        3.0f * std::fmaxf(S, K)
    );

    const float dS = (S_max - S_min) / static_cast<float>(N_s);

    // Stock price grid
    std::vector<float> S_grid(N_s + 1);
    for (int i = 0; i <= N_s; ++i) {
        S_grid[i] = S_min + i * dS;
    }

    // Build dividend schedule
    struct DivEvent {
        int step;
        float amount;
        float time;
    };
    std::vector<DivEvent> div_events;
    for (std::size_t k = 0; k < div_amounts.size() && k < div_times.size(); ++k) {
        float tD = div_times[k];
        if (tD > 0.0f && tD < T) {
            int step = static_cast<int>(std::floorf(tD / dt));
            if (step < 0) step = 0;
            if (step >= N_t) step = N_t - 1;
            div_events.push_back({step, div_amounts[k], tD});
        }
    }

    // Terminal condition
    std::vector<float> V(N_s + 1);
    for (int i = 0; i <= N_s; ++i) {
        float payoff = is_call
            ? std::fmaxf(S_grid[i] - K, 0.0f)
            : std::fmaxf(K - S_grid[i], 0.0f);
        V[i] = payoff;
    }

    // FD coefficients
    std::vector<float> a(N_s + 1), b(N_s + 1), c(N_s + 1);
    std::vector<float> rhs(N_s + 1);

    // Backward time stepping
    for (int n = N_t - 1; n >= 0; --n) {
        float t = n * dt;
        float r = get_rate_at_time_convex_monotone_host(t, rc, rt, n_rates);

        // Crank-Nicolson with theta = 0.5
        const float theta = 0.5f;
        const float sigma2 = sigma * sigma;

        // Interior points: i = 1 to N_s - 1
        for (int i = 1; i < N_s; ++i) {
            float Si = S_grid[i];
            if (Si < 1e-8f) {
                // Do NOT skip: define a stable identity row to avoid stale coefficients.
                a[i] = 0.0f;
                b[i] = 1.0f;
                c[i] = 0.0f;
                rhs[i] = V[i];
                continue;
            }

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

            if (is_call) {
                rhs[0] = 0.0f;
            } else {
                // Put at S=0: worth K * D(t,T)
                rhs[0] = K * df_t_T_host(t, T, rc, rt, n_rates);
            }
        }

        // Upper boundary: S = S_max
        {
            a[N_s] = 0.0f;
            b[N_s] = 1.0f;
            c[N_s] = 0.0f;

            if (is_call) {
                // Call at large S: V ≈ S - K * D(t,T)
                rhs[N_s] = S_grid[N_s] - K * df_t_T_host(t, T, rc, rt, n_rates);
            } else {
                // Put at large S: worth approximately 0
                rhs[N_s] = 0.0f;
            }
        }

        // Solve tridiagonal
        solve_tridiagonal(N_s + 1, a, b, c, rhs);
        V = rhs;

        // American early exercise at time t (before dividend)
        for (int i = 0; i <= N_s; ++i) {
            float intrinsic = is_call
                ? std::fmaxf(S_grid[i] - K, 0.0f)
                : std::fmaxf(K - S_grid[i], 0.0f);
            V[i] = std::fmaxf(V[i], intrinsic);
        }

        // Apply dividend jumps
        for (const auto& div : div_events) {
            if (div.step == n) {
                std::vector<float> V_new(N_s + 1);

                for (int i = 0; i <= N_s; ++i) {
                    float S_pre_div = S_grid[i];
                    float S_post_div = S_pre_div - div.amount;

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
                        int idx = static_cast<int>(std::floorf(idx_f));
                        if (idx < 0) idx = 0;
                        if (idx >= N_s) idx = N_s - 1;
                        float w = idx_f - idx;
                        V_new[i] = (1.0f - w) * V[idx] + w * V[idx + 1];
                    }
                }
                V = V_new;

                // Re-apply early exercise after dividend
                for (int i = 0; i <= N_s; ++i) {
                    float intrinsic = is_call
                        ? std::fmaxf(S_grid[i] - K, 0.0f)
                        : std::fmaxf(K - S_grid[i], 0.0f);
                    V[i] = std::fmaxf(V[i], intrinsic);
                }
            }
        }
    }

    // Final interpolation at spot S
    if (S <= S_min) return V[0];
    if (S >= S_max) return V[N_s];

    float idx_f = (S - S_min) / dS;
    int idx = static_cast<int>(std::floorf(idx_f));
    if (idx < 0) idx = 0;
    if (idx >= N_s) idx = N_s - 1;
    float w = idx_f - idx;

    return (1.0f - w) * V[idx] + w * V[idx + 1];
}

// Vectorized CPU FD price wrapper (American w/ discrete cash dividends)
std::vector<float> v_fd_price_host(
    const std::vector<float>& spots,
    const std::vector<float>& strikes,
    const std::vector<float>& tenors,
    const std::vector<float>& sigmas,
    const std::vector<uint8_t>& v_is_call,
    const std::vector<float>& rates_curve = {},
    const std::vector<float>& rates_times = {},
    const std::vector<float>& div_amounts = {},
    const std::vector<float>& div_times   = {},
    const int time_steps  = 200,
    const int space_steps = 200)
{
    const std::size_t n = spots.size();
    std::vector out(n, std::numeric_limits<float>::quiet_NaN());

    if (n == 0) return out;

    // Validate input sizes
    if (strikes.size() != n || tenors.size() != n || sigmas.size() != n || v_is_call.size() != n) {
        return out;
    }

    for (std::size_t i = 0; i < n; ++i) {
        const float   S       = spots[i];
        const float   K       = strikes[i];
        const float   T       = tenors[i];
        const float   sigma   = sigmas[i];
        const uint8_t is_call = v_is_call[i];

        // Basic sanity: match scalar behavior (intrinsic for invalid/degenerate inputs)
        if (!std::isfinite(S) || !std::isfinite(K) || !std::isfinite(T) || !std::isfinite(sigma)) {
            out[i] = std::numeric_limits<float>::quiet_NaN();
            continue;
        }

        out[i] = price_american_fd_div_host(
            S, K, T, sigma, is_call,
            rates_curve, rates_times,
            div_amounts, div_times,
            time_steps, space_steps
        );
    }

    return out;
}

// Refactored IV solver with better convergence
float implied_vol_american_fd_host(
    const float target,
    const float S,
    const float K,
    const float T,
    const uint8_t is_call,
    const std::vector<float>& rates_curve,
    const std::vector<float>& rates_times,
    const std::vector<float>& div_amounts,
    const std::vector<float>& div_times,
    const float tol         = 1e-6f,
    const int   max_iter    = 100,
    const float v_min       = 1e-4f,
    const float v_max       = 5.0f,
    const int   time_steps  = 200,
    const int   space_steps = 200
    )
{
    if (T <= 0.0f || S <= 0.0f || K <= 0.0f || target <= 0.0f)
        return 0.0f;

    // Check if price is at intrinsic
    float intrinsic = is_call ? std::fmax(S - K, 0.0f) : std::fmax(K - S, 0.0f);
    if (std::fabs(target - intrinsic) < 1e-6f) {
        return 0.0f;
    }

    // // For deep ITM puts with dividends, use tighter bounds
    // bool is_deep_itm_put = (!is_call) && (K > S * 1.2f);
    // if (is_deep_itm_put && !div_amounts.empty()) {
    //     v_max = std::min(v_max, 2.0f); // Caps at 200% vol
    // }

    // Initial bracket check
    float p_low = price_american_fd_div_host(
        S, K, T, v_min, is_call,
        rates_curve, rates_times,
        div_amounts, div_times,
        time_steps, space_steps);

    float p_high = price_american_fd_div_host(
        S, K, T, v_max, is_call,
        rates_curve, rates_times,
        div_amounts, div_times,
        time_steps, space_steps);

    // Target below minimum possible price. That can short circuit certain evaluations.
    // Would want to rerun with higher time / space steps
    if (target <= p_low + 1e-9f)
        return v_min;

    // Target above maximum possible price
    if (target >= p_high - 1e-9f)
        return std::numeric_limits<float>::quiet_NaN();

    // Bisection with adaptive tolerance
    float lo = v_min;
    float hi = v_max;
    float mid = 0.0f;

    // Use relative tolerance for better convergence
    float rel_tol = tol / std::fmax(target, 1.0f);

    for (int i = 0; i < max_iter; ++i) {
        mid = 0.5f * (lo + hi);

        float p_mid = price_american_fd_div_host(
            S, K, T, mid, is_call,
            rates_curve, rates_times,
            div_amounts, div_times,
            time_steps, space_steps);

        float diff = p_mid - target;
        float rel_diff = std::fabs(diff) / target;

        // Check convergence with both absolute and relative tolerance
        if (std::fabs(diff) < tol || rel_diff < rel_tol) {
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

// Refactored vectorized CPU FD IV
std::vector<float> get_v_iv_fd_cpu(
    const std::vector<float>& prices,
    const std::vector<float>& spots,
    const std::vector<float>& strikes,
    const std::vector<float>& tenors,
    const std::vector<uint8_t>& v_is_call,
    const std::vector<float>& rates_curve = {},
    const std::vector<float>& rates_times = {},
    const std::vector<float>& div_amounts = {},
    const std::vector<float>& div_times   = {},
    const float tol         = 1e-6f,
    const int   max_iter    = 100,
    const float v_min       = 1e-4f,
    const float v_max       = 5.0f,
    const int time_steps = 1000,
    const int space_steps = 1000
    )
{
    const std::size_t n_options = prices.size();
    std::vector results(n_options, std::numeric_limits<float>::quiet_NaN());

    if (n_options == 0)
        return results;

    // Validate input sizes
    if (spots.size()      != n_options ||
        strikes.size()    != n_options ||
        tenors.size()     != n_options ||
        v_is_call.size()  != n_options) {
        return results;
    }

    // Process each option
    for (std::size_t i = 0; i < n_options; ++i) {
        float p  = prices[i];
        float S  = spots[i];
        float K  = strikes[i];
        float T  = tenors[i];
        uint8_t is_call = v_is_call[i];

        // Validate inputs
        if (!std::isfinite(p) || !std::isfinite(S) ||
            !std::isfinite(K) || !std::isfinite(T)) {
            results[i] = std::numeric_limits<float>::quiet_NaN();
            continue;
        }

        if (T <= 0.0f || S <= 0.0f || K <= 0.0f || p <= 0.0f) {
            results[i] = (T <= 0.0f) ? 0.0f : std::numeric_limits<float>::quiet_NaN();
            continue;
        }

        // Quick intrinsic check
        float intrinsic = is_call ? std::fmax(S - K, 0.0f) : std::fmax(K - S, 0.0f);
        if (std::fabs(p - intrinsic) < 1e-6f) {
            results[i] = 0.0f;
            continue;
        }

        // Check if price is reasonable (not below intrinsic)
        if (p < intrinsic - 1e-6f) {
            results[i] = v_min;
            continue;
        }

        // Solve for IV
        results[i] = implied_vol_american_fd_host(
            p, S, K, T, is_call,
            rates_curve, rates_times,
            div_amounts, div_times,
            tol, max_iter, v_min, v_max,
            time_steps, space_steps);
    }

    return results;
}


// ---- NEW: public wrappers for tests/diagnostics (no duplicated math) ----
float merlin_df_0_t_host(
    float t,
    const std::vector<float>& rates_curve,
    const std::vector<float>& rates_times)
{
    const float* rc = rates_curve.empty() ? nullptr : rates_curve.data();
    const float* rt = rates_times.empty() ? nullptr : rates_times.data();
    const int n_rates = static_cast<int>(rates_curve.size());
    return df_0_t_host(t, rc, rt, n_rates);
}

float merlin_df_t_T_host(
    float t,
    float T,
    const std::vector<float>& rates_curve,
    const std::vector<float>& rates_times)
{
    const float* rc = rates_curve.empty() ? nullptr : rates_curve.data();
    const float* rt = rates_times.empty() ? nullptr : rates_times.data();
    const int n_rates = static_cast<int>(rates_curve.size());
    return df_t_T_host(t, T, rc, rt, n_rates);
}

float merlin_forward_rate_host(
    float t,
    const std::vector<float>& rates_curve,
    const std::vector<float>& rates_times)
{
    const float* rc = rates_curve.empty() ? nullptr : rates_curve.data();
    const float* rt = rates_times.empty() ? nullptr : rates_times.data();
    const int n_rates = static_cast<int>(rates_curve.size());
    return get_rate_at_time_host(t, rc, rt, n_rates);
}

float merlin_forward_rate_convex_monotone_host(
    float t,
    const std::vector<float>& rates_curve,
    const std::vector<float>& rates_times)
{
    const float* rc = rates_curve.empty() ? nullptr : rates_curve.data();
    const float* rt = rates_times.empty() ? nullptr : rates_times.data();
    const int n_rates = static_cast<int>(rates_curve.size());
    return get_rate_at_time_convex_monotone_host(t, rc, rt, n_rates);
}