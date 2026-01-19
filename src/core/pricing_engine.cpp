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


// Improved FD pricer with better put handling near dividends
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

    // Calculate PV of dividends for grid construction
    float pv_divs = 0.0f;
    for (std::size_t i = 0; i < div_amounts.size() && i < div_times.size(); ++i) {
        float t_div = div_times[i];
        if (t_div > 0.0f && t_div <= T) {
            float r_div = get_rate_at_time_host(t_div,
                                               rates_curve.empty() ? nullptr : rates_curve.data(),
                                               rates_times.empty() ? nullptr : rates_times.data(),
                                               static_cast<int>(rates_curve.size()));
            pv_divs += div_amounts[i] * std::exp(-r_div * t_div);
        }
    }

    // float S_grid_center = std::fmax(S - pv_divs, S * 0.1f);

    // Build spatial grid - ensure it covers likely dividend scenarios
    float r0 = get_rate_at_time_host(0.0f,
                                     rates_curve.empty() ? nullptr : rates_curve.data(),
                                     rates_times.empty() ? nullptr : rates_times.data(),
                                     static_cast<int>(rates_curve.size()));

    // For puts, ensure grid goes low enough to capture deep ITM scenarios
    float S_min = 0.0f;  // Always start at zero for puts
    float S_max = std::fmaxf(
        (S + pv_divs) * std::expf((r0 + 4.0f * sigma) * T),
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
        float r = get_rate_at_time_host(t,
                                        rates_curve.empty() ? nullptr : rates_curve.data(),
                                        rates_times.empty() ? nullptr : rates_times.data(),
                                        static_cast<int>(rates_curve.size()));

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
                rhs[0] = K * std::expf(-r * t_remaining);
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
                rhs[N_s] = S_grid[N_s] - K * std::expf(-r * t_remaining);
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

    // Target below minimum possible price
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
    const int time_steps = 200,
    const int space_steps = 200
    )
{
    const std::size_t n_options = prices.size();
    std::vector<float> results(n_options, std::numeric_limits<float>::quiet_NaN());

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
            results[i] = std::numeric_limits<float>::quiet_NaN();
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

// static float price_american_binomial_cash_div_threadlocal_host(
//     float S, float K, float T,
//     float sigma, uint8_t is_call, int n_steps,
//     const float *rates_curve, const float *rates_times, int n_rates,
//     const float *div_amounts, const int *div_steps, int n_divs) {
//
//     constexpr int MAX_STEPS = 512;
//     float V[MAX_STEPS + 1];
//
//     // Defensive bound for exponent
//     if (n_steps > MAX_STEPS) n_steps = MAX_STEPS;
//
//     // Scaling time steps for short T (same as device)
//     int effective_steps = n_steps;
//     if (T < 0.1f) {
//         effective_steps = static_cast<int>(T * n_steps * 10);
//         if (effective_steps > n_steps)  effective_steps = n_steps;
//         if (effective_steps < 10)       effective_steps = 10;
//     }
//     if (effective_steps > MAX_STEPS) effective_steps = MAX_STEPS;
//
//     float dt = T / effective_steps;
//
//     // Same dt sanity shortcuts
//     if (dt <= 0.0f) {
//         return is_call ? std::fmax(S - K, 0.0f) : std::fmax(K - S, 0.0f);
//     }
//     if (dt < 1e-6f) {
//         return is_call ? std::fmax(S - K, 0.0f) : std::fmax(K - S, 0.0f);
//     }
//
//     // Standard CRR multipliers
//     float u = std::exp(sigma * std::sqrt(dt));
//     if (u <= 1.001f) u = 1.001f;  // numerical safety
//     float d = 1.0f / u;
//     int   N = effective_steps;
//
//     // 1. Terminal payoffs with cash dividends applied along each path
//     float u2     = u * u;
//     float curr_S = S * std::pow(d, static_cast<float>(N));  // bottom node (all downs)
//
//     for (int j = 0; j <= N; ++j) {
//         float S_j = curr_S;
//
//         // Apply each dividend along this path
//         if (n_divs > 0) {
//             for (int div_i = 0; div_i < n_divs; ++div_i) {
//                 int step_div = div_steps[div_i];
//                 if (step_div > 0 && step_div <= N) {
//                     int ups_before_div   = (step_div <= j) ? step_div : j;
//                     int downs_before_div = step_div - ups_before_div;
//
//                     int ups_after_div   = j - ups_before_div;
//                     int downs_after_div = (N - j) - downs_before_div;
//
//                     float mult_after =
//                         std::pow(u, static_cast<float>(ups_after_div)) *
//                         std::pow(d, static_cast<float>(downs_after_div));
//
//                     S_j -= div_amounts[div_i] * mult_after;
//                     if (S_j < 1e-8f) S_j = 1e-8f;
//                 }
//             }
//         }
//
//         V[j] = is_call ? std::fmax(S_j - K, 0.0f) : std::fmax(K - S_j, 0.0f);
//         curr_S *= u2;  // move to next node: replace one d with one u
//     }
//
//     // 2. Backward induction with term structure and discrete dividends
//     for (int step = N - 1; step >= 0; --step) {
//         float t_curr = step * dt;
//         float r      = get_rate_at_time_host(t_curr, rates_curve, rates_times, n_rates);
//         float disc   = std::exp(-r * dt);
//         float edtr   = std::exp(r * dt);
//         float p      = (edtr - d) / (u - d);
//         // clamp to [0,1]
//         if (p < 0.0f) p = 0.0f;
//         if (p > 1.0f) p = 1.0f;
//
//         for (int j = 0; j <= step; ++j) {
//             float cont = disc * (p * V[j + 1] + (1.0f - p) * V[j]);
//
//             // Rebuild stock at node (step, j) ignoring dividends
//             float S_j =
//                 S *
//                 std::pow(u, static_cast<float>(j)) *
//                 std::pow(d, static_cast<float>(step - j));
//
//             // Apply all dividends that have occurred up to and including this step
//             if (n_divs > 0) {
//                 for (int div_i = 0; div_i < n_divs; ++div_i) {
//                     int step_div = div_steps[div_i];
//                     if (step_div <= step && step_div > 0) {
//                         int ups_before_div   = (step_div <= j) ? step_div : j;
//                         int downs_before_div = step_div - ups_before_div;
//
//                         int ups_after_div   = j - ups_before_div;
//                         int downs_after_div = (step - j) - downs_before_div;
//
//                         float mult_after =
//                             std::pow(u, static_cast<float>(ups_after_div)) *
//                             std::pow(d, static_cast<float>(downs_after_div));
//
//                         S_j -= div_amounts[div_i] * mult_after;
//                         if (S_j < 1e-8f) S_j = 1e-8f;
//                     }
//                 }
//             }
//
//             float ex = is_call ? std::fmax(S_j - K, 0.0f) : std::fmax(K - S_j, 0.0f);
//             V[j] = std::fmax(cont, ex);
//         }
//     }
//     return V[0];
// }

// float implied_vol_american_bisection_threadlocal_host(
//     float target, float S, float K, float T, uint8_t is_call, int n_steps,
//     const float *rates_curve, const float *rates_times, int n_rates,
//     const float *div_amounts, const int *div_steps, int n_divs,
//     float tol = 1e-6f, int max_iter = 100, float v_min = 1e-4f, float v_max = 5.0f) {
//     float p_low = price_american_binomial_cash_div_threadlocal_host(S, K, T, v_min, is_call, n_steps, rates_curve,
//                                                                rates_times, n_rates, div_amounts, div_steps, n_divs);
//     float p_high = price_american_binomial_cash_div_threadlocal_host(S, K, T, v_max, is_call, n_steps, rates_curve,
//                                                                 rates_times, n_rates, div_amounts, div_steps, n_divs);
//     if (target <= p_low + 1e-12f) return 0.0f;
//     if (target > p_high + 1e-12f) return NAN;
//
//     float lo = v_min, hi = v_max, mid = 0.0f;
//     for (int i = 0; i < max_iter; ++i) {
//         mid = 0.5f * (lo + hi);
//         float p_mid = price_american_binomial_cash_div_threadlocal_host(S, K, T, mid, is_call, n_steps, rates_curve,
//                                                                    rates_times, n_rates, div_amounts, div_steps,
//                                                                    n_divs);
//         float diff = p_mid - target;
//         if (fabsf(diff) < tol) return mid;
//         if (diff < 0.0f) lo = mid;
//         else hi = mid;
//         if (hi - lo < tol) break;
//     }
//     return mid;
// }

// float get_single_iv_cpu(
//     float target_price, float S, float K, float T, uint8_t is_call,
//     const std::vector<float> &rates_curve, const std::vector<float> &rates_times,
//     const std::vector<float> &div_amounts, const std::vector<float> &div_times,
//     float tol, int max_iter, const float steps_factor) {
//
//     if (T <= 0.0f || S <= 0.0f || K <= 0.0f || target_price <= 0.0f ||
//         !std::isfinite(target_price) || !std::isfinite(S) ||
//         !std::isfinite(K) || !std::isfinite(T)) {
//         return (T <= 0.0f) ? 0.0f : std::numeric_limits<float>::quiet_NaN();
//     }
//
//     // Match the GPU IV kernel convention: n_steps = 200 * steps_factor
//     int n_steps = static_cast<int>(200 * steps_factor);
//     if (n_steps < 1) n_steps = 1;
//
//     // Prepare rate pointers for the host pricer
//     const float *rates_curve_ptr = rates_curve.empty() ? nullptr : rates_curve.data();
//     const float *rates_times_ptr = rates_times.empty() ? nullptr : rates_times.data();
//     int n_rates = static_cast<int>(rates_curve.size());
//
//     // Map dividend times to integer steps exactly as compute_iv_kernel_threadlocal does
//     const int MAX_DIVS = 32;
//     float local_div_amounts[MAX_DIVS];
//     int   local_div_steps[MAX_DIVS];
//     int   valid_divs = 0;
//
//     int n_divs = static_cast<int>(div_amounts.size());
//     for (int j = 0; j < n_divs && j < MAX_DIVS; ++j) {
//         float div_time = div_times[j];
//         if (div_time > 0.0f && div_time < T) {
//             int step = (int) roundf(div_time / T * n_steps);
//             step = std::max(1, std::min(step, n_steps));
//             local_div_steps[valid_divs] = step;
//             local_div_amounts[valid_divs] = div_amounts[j];
//             valid_divs++;
//         }
//     }
//
//     return implied_vol_american_bisection_threadlocal_host(target_price, S, K, T, is_call, n_steps,
//     rates_curve_ptr, rates_times_ptr, n_rates,
//     local_div_amounts, local_div_steps, valid_divs
//     );
// }
