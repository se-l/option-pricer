
__device__ float price_american_fd_div_cuda(
    const float S, const float K, const float T,
    const float sigma, const uint8_t is_call,
    const float* rates_curve,
    const float* rates_times,
    const int n_rates,
    const float pv_divs,
    const DivEvent* div_events,
    const int n_div_events,
    const int time_steps = 200,
    const int space_steps = 200
    )
{
    // Keep these caps close to what you actually use (e.g., 200).
    // This reduces per-thread stack usage and local-memory spilling.
    constexpr int MAX_TIME_STEPS = 256;
    constexpr int MAX_SPACE_STEPS = 256;

    if (T <= 0.0f || S <= 0.0f || K <= 0.0f || sigma <= 0.0f)
        return is_call ? fmaxf(S - K, 0.0f) : fmaxf(K - S, 0.0f);

    // Bound the grid sizes
    const int N_t = (time_steps < 10) ? 10 : ((time_steps > MAX_TIME_STEPS) ? MAX_TIME_STEPS : time_steps);
    const int N_s = (space_steps < 10) ? 10 : ((space_steps > MAX_SPACE_STEPS) ? MAX_SPACE_STEPS : space_steps);

    const float dt = T / static_cast<float>(N_t);  // precalc given only have ~20 tenors

    const float r0 = get_rate_at_time_convex_monotone(0.0f, rates_curve, rates_times, n_rates);  // precalc given only have 1 calc_date

    // For puts, ensure grid goes low enough to capture deep ITM scenarios
    constexpr float S_min = 0.0f;  // Always start at zero for puts
    const float S_max = fmaxf(
        (S + pv_divs) * expf(fminf((r0 + 4.0f * sigma) * T, 80.0f)),
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

    // Allocate once (not inside dividend loop)
    float V_new[MAX_SPACE_STEPS + 1];

    for (int i = 0; i <= N_s; ++i) {
        float payoff = is_call
            ? fmaxf(S_grid[i] - K, 0.0f)
            : fmaxf(K - S_grid[i], 0.0f);
        V[i] = payoff;
    }

    // Backward time stepping
    for (int n = N_t - 1; n >= 0; --n) {
        float t = n * dt;
        float r = get_rate_at_time_convex_monotone(t, rates_curve, rates_times, n_rates);  // precalc given only ~20 tenors..

        // Crank-Nicolson with theta = 0.5
        const float theta = 0.5f;
        const float sigma2 = sigma * sigma;

        // Interior points: i = 1 to N_s - 1
        for (int i = 1; i < N_s; ++i) {
            const float Si = S_grid[i];
            if (Si < 1e-8f) {
                // Do NOT skip: define a stable identity row to avoid stale coefficients.
                a[i] = 0.0f;
                b[i] = 1.0f;
                c[i] = 0.0f;
                rhs[i] = V[i];
                continue;
            }

            const float Si2 = Si * Si;

            // Standard FD coefficients in price space
            const float coef_pp = 0.5f * sigma2 * Si2 / (dS * dS);  // S²σ²/2
            const float coef_p = 0.5f * r * Si / dS;                 // rS/2
            const float coef_0 = -sigma2 * Si2 / (dS * dS) - r;     // -S²σ² - r

            const float alpha = coef_pp - coef_p;   // coefficient of V[i-1]
            const float beta = coef_0;               // coefficient of V[i]
            const float gamma = coef_pp + coef_p;    // coefficient of V[i+1]

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
                // Put at S=0: worth K * D(t,T) under term structure
                rhs[0] = K * df_t_T(t, T, rates_curve, rates_times, n_rates);
            }
        }

        // Upper boundary: S = S_max
        {
            a[N_s] = 0.0f;
            b[N_s] = 1.0f;
            c[N_s] = 0.0f;

            if (is_call) {
                // Call at large S: V ≈ S - K * D(t,T)
                rhs[N_s] = S_grid[N_s] - K * df_t_T(t, T, rates_curve, rates_times, n_rates);
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
            const float intrinsic = is_call
                ? fmaxf(S_grid[i] - K, 0.0f)
                : fmaxf(K - S_grid[i], 0.0f);
            V[i] = fmaxf(V[i], intrinsic);
        }

        // Apply dividend jumps
        for (int div_idx = 0; div_idx < n_div_events; ++div_idx) {
            if (div_events[div_idx].step == n) {
                // float V_new[MAX_SPACE_STEPS + 1];

                for (int i = 0; i <= N_s; ++i) {
                    const float S_pre_div = S_grid[i];
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
                for (int i = 0; i <= N_s; ++i) {
                    V[i] = V_new[i];
                }

                // Re-apply early exercise after dividend
                for (int i = 0; i <= N_s; ++i) {
                    const float intrinsic = is_call
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

    const float idx_f = (S - S_min) / dS;
    int idx = static_cast<int>(floorf(idx_f));
    if (idx < 0) idx = 0;
    if (idx >= N_s) idx = N_s - 1;
    const float w = idx_f - idx;

    return (1.0f - w) * V[idx] + w * V[idx + 1];
}

// NEW kernel: reuses per-tenor precomputed dividend schedule (no precompute_dividends_cuda),
// and uses sigma_guess to tighten the IV bracket.
__global__ void compute_fd_iv_kernel_with_guess_tenor_cache(
    const float *prices, const float *spots, const float *strikes, const float *tenors,
    const uint8_t *v_is_call, const int n_options,
    const int *tenor_ids,
    const float *pv_divs_by_tenor,
    const DivEvent *div_events_by_tenor,
    const int *n_div_events_by_tenor,
    const float *rates_curve, const float *rates_times, const int n_rates,
    const float *sigma_guess, // may be nullptr
    float *results,
    const float price_tol, const int max_iter, const float v_min, const float v_max,
    const float iv_eps,
    const float guess_width,
    const int time_steps, const int space_steps)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_options) return;

    const float p = prices[idx];
    const float s = spots[idx];
    const float k = strikes[idx];
    const float t = tenors[idx];
    const uint8_t is_call = v_is_call[idx];

    if (!isfinite(p) || !isfinite(s) || !isfinite(k) || !isfinite(t)) {
        results[idx] = NAN;
        return;
    }
    if (t <= 0.0f || s <= 0.0f || k <= 0.0f || p <= 0.0f) {
        results[idx] = (t <= 0.0f) ? 0.0f : NAN;
        return;
    }

    float intrinsic = is_call ? fmaxf(s - k, 0.0f) : fmaxf(k - s, 0.0f);
    if (fabsf(p - intrinsic) < 1e-6f) {
        results[idx] = 0.0f;
        return;
    }
    if (p < intrinsic - 1e-6f) {
        results[idx] = NAN;
        return;
    }

    int tid = tenor_ids[idx];
    float pv_divs = pv_divs_by_tenor[tid];
    const DivEvent* div_events = div_events_by_tenor + tid * MAX_DIV_EVENTS;
    int n_div_events = n_div_events_by_tenor[tid];

    float g = (sigma_guess != nullptr) ? sigma_guess[idx] : NAN;
    float lo_init = v_min;
    float hi_init = v_max;

    if (isfinite(g) && g > 0.0f) {
        float w = fmaxf(0.0f, guess_width);
        lo_init = fmaxf(v_min, (1.0f - w) * g);
        hi_init = fminf(v_max, (1.0f + w) * g);
        if (!(hi_init > lo_init)) {
            lo_init = v_min;
            hi_init = v_max;
        }
    }

    results[idx] = implied_vol_american_fd_cuda_bracketed(
        p, s, k, t, is_call,
        rates_curve, rates_times, n_rates,
        pv_divs, div_events, n_div_events,
        lo_init, hi_init,
        price_tol, max_iter, v_min, v_max,
        iv_eps,
        time_steps, space_steps
    );
}

float median(std::vector<float> v) {
    const size_t n = v.size();
    const size_t mid = n / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    float hi = v[mid];
    if (n % 2) return hi;
    std::nth_element(v.begin(), v.begin() + (mid - 1), v.begin() + mid);
    float lo = v[mid - 1];
    return 0.5f * (lo + hi);
}

__global__ void normalize_prices_kernel(const float* in, float* out, int n, float invS) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = in[i] * invS;
}

// ------------------------------
// Sigma grid for cache (constant memory)
// ------------------------------
constexpr int MAX_SIGMA_GRID = 64;
__constant__ float c_sigma_grid[MAX_SIGMA_GRID];

// ------------------------------
// Log-moneyness cache configuration
// ------------------------------
// You should tune these to your universe.
// Example: K/S from ~0.5..2.0  => log-m in [-0.693..0.693]
constexpr int LM_BINS = 128;
constexpr float LM_MIN = -1.0f;   // ~K/S=0.3679
constexpr float LM_MAX =  1.0f;   // ~K/S=2.7183

// ------------------------------
// GPU precompute: FD price cache for (right, tenor_id, lm_bin) x sigma_grid
// ------------------------------
// We precompute on a normalized spot S0=1.0 and set K = exp(lm_center) * S0.
// (This gives you a robust initial IV guess across varying S by conditioning on log-moneyness.)
__global__ void compute_fd_price_cache_lm_kernel(
    const float S0,
    const float* lm_centers, int n_lm,
    const float* tenors_unique, int n_tenors,
    const uint8_t* rights_unique, int n_rights,
    const float* rates_curve, const float* rates_times, int n_rates,
    const float* pv_divs_by_tenor,
    const DivEvent* div_events_by_tenor,
    const int* n_div_events_by_tenor,
    const int time_steps, const int space_steps,
    const int n_sigma,
    float* out_prices) // size = (n_tenors*n_lm*n_rights) * n_sigma
{
    const int tenor_id = blockIdx.x;
    const int lm_id    = blockIdx.y;
    const int right_id = blockIdx.z;
    const int j        = threadIdx.x;

    if (tenor_id >= n_tenors || lm_id >= n_lm || right_id >= n_rights) return;
    if (j >= n_sigma) return;

    const float T = tenors_unique[tenor_id];
    const float lm = lm_centers[lm_id];
    const float K = expf(lm) * S0;
    const uint8_t right = rights_unique[right_id];

    const int bucket_id = ((tenor_id * n_lm) + lm_id) * n_rights + right_id;

    const float pv_divs = pv_divs_by_tenor[tenor_id];
    const DivEvent* div_events = div_events_by_tenor + tenor_id * MAX_DIV_EVENTS;
    const int n_div_events = n_div_events_by_tenor[tenor_id];

    const float sigma = c_sigma_grid[j];

    out_prices[bucket_id * n_sigma + j] = price_american_fd_div_cuda(
        S0, K, T, sigma, right,
        rates_curve, rates_times, n_rates,
        pv_divs, div_events, n_div_events,
        time_steps, space_steps
    );
}

// Compute per-option sigma guess using log-moneyness bin and cached sigma->price curve.
__global__ void compute_sigma_guess_from_lm_cache_kernel(
    const float* prices, const float* spots, const float* strikes, const float* tenors, const uint8_t* rights,
    const int* tenor_ids, const int n_options,
    const int n_lm, const int n_rights,
    const int n_sigma,
    const float* price_cache, // [n_buckets * n_sigma]
    float* out_sigma_guess)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_options) return;

    const float p = prices[idx];
    const float S = spots[idx];
    const float K = strikes[idx];
    const int tid = tenor_ids[idx];
    const int rid = static_cast<int>(rights[idx]); // assumes 0/1 matches rights_unique ordering

    if (!isfinite(p) || !isfinite(S) || !isfinite(K) || S <= 0.0f || K <= 0.0f) {
        out_sigma_guess[idx] = NAN;
        return;
    }

    float lm = logf(K / S);
    // clamp to cache domain
    lm = fminf(fmaxf(lm, LM_MIN), LM_MAX);

    const float u = (lm - LM_MIN) / (LM_MAX - LM_MIN);
    int lm_id = static_cast<int>(floorf(u * static_cast<float>(n_lm)));
    if (lm_id < 0) lm_id = 0;
    if (lm_id >= n_lm) lm_id = n_lm - 1;

    const int bucket_id = ((tid * n_lm) + lm_id) * n_rights + rid;
    const float* curve = price_cache + bucket_id * n_sigma;

    const float p0 = curve[0];
    const float pN = curve[n_sigma - 1];

    if (!isfinite(p0) || !isfinite(pN)) {
        out_sigma_guess[idx] = NAN;
        return;
    }
    if (p <= p0) {
        out_sigma_guess[idx] = c_sigma_grid[0];
        return;
    }
    if (p >= pN) {
        out_sigma_guess[idx] = c_sigma_grid[n_sigma - 1];
        return;
    }

    // linear scan (n_sigma <= 64) is cache-friendly and avoids branchy binary search
    int j0 = 0;
    for (int j = 0; j < n_sigma - 1; ++j) {
        float a = curve[j];
        float b = curve[j + 1];
        if (p >= a && p <= b) { j0 = j; break; }
    }

    const float P_lo = curve[j0];
    const float P_hi = curve[j0 + 1];
    const float s_lo = c_sigma_grid[j0];
    const float s_hi = c_sigma_grid[j0 + 1];

    const float denom = (P_hi - P_lo);
    float w = (fabsf(denom) < 1e-12f) ? 0.0f : (p - P_lo) / denom;
    w = fminf(fmaxf(w, 0.0f), 1.0f);

    out_sigma_guess[idx] = s_lo + w * (s_hi - s_lo);
}
// NEW: build LM-cache on GPU and produce per-option sigma guess (device pointer).
// Caller owns returned device pointer and must cudaFree() it.
static float* build_sigma_guess_from_lm_cache_cuda(
    const int n_options,
    const float* d_prices_abs,   // absolute option prices on device
    const float* d_spots,
    const float* d_strikes,
    const float* d_tenors,
    const uint8_t* d_rights,
    const int* d_tenor_ids,
    const int n_tenors,
    const float* d_tenors_unique,
    const float* d_rates_curve,
    const float* d_rates_times,
    const int n_rates,
    const float* d_pv_divs_by_tenor_norm,          // normalized by S_ref
    const DivEvent* d_div_events_by_tenor_norm,    // amounts normalized by S_ref
    const int* d_n_div_events_by_tenor,
    const int time_steps,
    const int space_steps,
    const int n_sigma,
    const float S_ref)
{
    // 1) Build lm_centers and copy to device
    std::vector<float> lm_centers(LM_BINS);
    for (int i = 0; i < LM_BINS; ++i) {
        float uu = (LM_BINS == 1) ? 0.0f : (static_cast<float>(i) / static_cast<float>(LM_BINS - 1));
        lm_centers[i] = LM_MIN + uu * (LM_MAX - LM_MIN);
    }

    float* d_lm_centers = nullptr;
    cudaMalloc(&d_lm_centers, LM_BINS * sizeof(float));
    cudaMemcpy(d_lm_centers, lm_centers.data(), LM_BINS * sizeof(float), cudaMemcpyHostToDevice);

    // 2) Rights unique {0,1}
    uint8_t rights_unique_host[2] = {0, 1};
    constexpr int n_rights = 2;
    uint8_t* d_rights_unique = nullptr;
    cudaMalloc(&d_rights_unique, n_rights * sizeof(uint8_t));
    cudaMemcpy(d_rights_unique, rights_unique_host, n_rights * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // 3) Cache compute at normalized spot
    const float S0_cache = 1.0f;
    const int n_buckets = n_tenors * LM_BINS * n_rights;

    float* d_price_cache = nullptr;
    cudaMalloc(&d_price_cache, static_cast<size_t>(n_buckets) * n_sigma * sizeof(float));

    dim3 grid_cache(n_tenors, LM_BINS, n_rights);
    dim3 block_cache(n_sigma, 1, 1);

    compute_fd_price_cache_lm_kernel<<<grid_cache, block_cache>>>(
        S0_cache,
        d_lm_centers, LM_BINS,
        d_tenors_unique, n_tenors,
        d_rights_unique, n_rights,
        d_rates_curve, d_rates_times, n_rates,
        d_pv_divs_by_tenor_norm,
        d_div_events_by_tenor_norm,
        d_n_div_events_by_tenor,
        time_steps, space_steps,
        n_sigma,
        d_price_cache
    );

    // 4) Normalize option prices for matching: p_norm = p_abs / S_ref
    float* d_prices_norm = nullptr;
    cudaMalloc(&d_prices_norm, n_options * sizeof(float));

    float invS = (std::isfinite(S_ref) && S_ref > 0.0f) ? (1.0f / S_ref) : 1.0f;
    {
        int block = 256;
        int grid = (n_options + block - 1) / block;
        normalize_prices_kernel<<<grid, block>>>(d_prices_abs, d_prices_norm, n_options, invS);
    }

    // 5) Sigma guess
    float* d_sigma_guess = nullptr;
    cudaMalloc(&d_sigma_guess, n_options * sizeof(float));
    {
        int block = 256;
        int grid = (n_options + block - 1) / block;
        compute_sigma_guess_from_lm_cache_kernel<<<grid, block>>>(
            d_prices_norm, d_spots, d_strikes, d_tenors, d_rights,
            d_tenor_ids, n_options,
            LM_BINS, n_rights,
            n_sigma,
            d_price_cache,
            d_sigma_guess
        );
    }

    // 6) Free temporaries (keep d_sigma_guess)
    cudaFree(d_prices_norm);
    cudaFree(d_price_cache);
    cudaFree(d_rights_unique);
    cudaFree(d_lm_centers);

    return d_sigma_guess;
}

std::vector<float> get_v_iv_fd_c_guessing_sigma(
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
    int n_options = static_cast<int>(prices.size());
    std::vector<float> results(n_options, std::numeric_limits<float>::quiet_NaN());
    if (n_options == 0) return results;

    // Validate input sizes
    if (spots.size() != static_cast<size_t>(n_options) ||
        strikes.size() != static_cast<size_t>(n_options) ||
        tenors.size() != static_cast<size_t>(n_options) ||
        v_is_call.size() != static_cast<size_t>(n_options)) {
        return results;
    }
    if (!rates_curve.empty() && rates_curve.size() != rates_times.size()) {
        return results;
    }
    if (!div_amounts.empty() && div_amounts.size() != div_times.size()) {
        return results;
    }

    // ---- 1) Upload sigma grid to constant memory (unchanged)
    constexpr int n_sigma = 48;
    {
        std::vector<float> sigma_grid(n_sigma);
        const float s0 = 1e-4f;
        const float s1 = 3.0f;
        const float log0 = logf(s0);
        const float log1 = logf(s1);
        for (int j = 0; j < n_sigma; ++j) {
            float uu = static_cast<float>(j) / static_cast<float>(n_sigma - 1);
            sigma_grid[j] = expf(log0 + uu * (log1 - log0));
        }
        cudaMemcpyToSymbol(c_sigma_grid, sigma_grid.data(), n_sigma * sizeof(float), 0, cudaMemcpyHostToDevice);
    }

    // ---- 2) Build two dividend schedules:
    // (a) ABSOLUTE for actual pricing/IV solve
    // (b) NORMALIZED by S_ref for cache/guess only
    const float S_ref = median(spots);
    const float div_scale_abs  = 1.0f;
    const float div_scale_norm = (std::isfinite(S_ref) && S_ref > 0.0f) ? (1.0f / S_ref) : 1.0f;

    TenorDividendScheduleHost sched_abs = build_tenor_dividend_schedule_host(
        tenors, time_steps, rates_curve, rates_times, div_amounts, div_times, div_scale_abs
    );
    TenorDividendScheduleHost sched_norm = build_tenor_dividend_schedule_host(
        tenors, time_steps, rates_curve, rates_times, div_amounts, div_times, div_scale_norm
    );

    const int n_tenors = static_cast<int>(sched_abs.unique_tenors.size());

    // ---- 3) Device allocations for core inputs (absolute)
    float *d_prices=nullptr, *d_spots=nullptr, *d_strikes=nullptr, *d_tenors=nullptr, *d_results=nullptr;
    uint8_t *d_rights=nullptr;

    int *d_tenor_ids=nullptr;
    float *d_tenors_unique=nullptr;

    float *d_rates_curve=nullptr, *d_rates_times=nullptr;

    // ABS schedule device arrays
    float *d_pv_divs_by_tenor_abs=nullptr;
    DivEvent *d_div_events_by_tenor_abs=nullptr;
    int *d_n_div_events_by_tenor_abs=nullptr;

    // NORM schedule device arrays (for cache/guess only)
    float *d_pv_divs_by_tenor_norm=nullptr;
    DivEvent *d_div_events_by_tenor_norm=nullptr;
    int *d_n_div_events_by_tenor_norm=nullptr;

    cudaMalloc(&d_prices, n_options * sizeof(float));
    cudaMalloc(&d_spots, n_options * sizeof(float));
    cudaMalloc(&d_strikes, n_options * sizeof(float));
    cudaMalloc(&d_tenors, n_options * sizeof(float));
    cudaMalloc(&d_rights, n_options * sizeof(uint8_t));
    cudaMalloc(&d_results, n_options * sizeof(float));

    cudaMemcpy(d_prices, prices.data(), n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_spots, spots.data(), n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_strikes, strikes.data(), n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tenors, tenors.data(), n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rights, v_is_call.data(), n_options * sizeof(uint8_t), cudaMemcpyHostToDevice);

    cudaMalloc(&d_tenor_ids, n_options * sizeof(int));
    cudaMemcpy(d_tenor_ids, sched_abs.tenor_ids.data(), n_options * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_tenors_unique, n_tenors * sizeof(float));
    cudaMemcpy(d_tenors_unique, sched_abs.unique_tenors.data(), n_tenors * sizeof(float), cudaMemcpyHostToDevice);

    // ABS schedule upload
    cudaMalloc(&d_pv_divs_by_tenor_abs, n_tenors * sizeof(float));
    cudaMalloc(&d_n_div_events_by_tenor_abs, n_tenors * sizeof(int));
    cudaMalloc(&d_div_events_by_tenor_abs, static_cast<size_t>(n_tenors) * MAX_DIV_EVENTS * sizeof(DivEvent));

    cudaMemcpy(d_pv_divs_by_tenor_abs, sched_abs.pv_divs_by_tenor.data(), n_tenors * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_n_div_events_by_tenor_abs, sched_abs.n_div_events_by_tenor.data(), n_tenors * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_div_events_by_tenor_abs, sched_abs.div_events_by_tenor.data(),
               static_cast<size_t>(n_tenors) * MAX_DIV_EVENTS * sizeof(DivEvent), cudaMemcpyHostToDevice);

    // NORM schedule upload (cache/guess only)
    cudaMalloc(&d_pv_divs_by_tenor_norm, n_tenors * sizeof(float));
    cudaMalloc(&d_n_div_events_by_tenor_norm, n_tenors * sizeof(int));
    cudaMalloc(&d_div_events_by_tenor_norm, static_cast<size_t>(n_tenors) * MAX_DIV_EVENTS * sizeof(DivEvent));

    cudaMemcpy(d_pv_divs_by_tenor_norm, sched_norm.pv_divs_by_tenor.data(), n_tenors * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_n_div_events_by_tenor_norm, sched_norm.n_div_events_by_tenor.data(), n_tenors * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_div_events_by_tenor_norm, sched_norm.div_events_by_tenor.data(),
               static_cast<size_t>(n_tenors) * MAX_DIV_EVENTS * sizeof(DivEvent), cudaMemcpyHostToDevice);

    const int n_rates = static_cast<int>(rates_curve.size());
    if (n_rates > 0) {
        cudaMalloc(&d_rates_curve, n_rates * sizeof(float));
        cudaMalloc(&d_rates_times, n_rates * sizeof(float));
        cudaMemcpy(d_rates_curve, rates_curve.data(), n_rates * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_rates_times, rates_times.data(), n_rates * sizeof(float), cudaMemcpyHostToDevice);
    }

    // ---- 4) Cache/guess phase (normalized dividends + normalized prices)
    float* d_sigma_guess = build_sigma_guess_from_lm_cache_cuda(
        n_options,
        d_prices, d_spots, d_strikes, d_tenors, d_rights,
        d_tenor_ids,
        n_tenors,
        d_tenors_unique,
        d_rates_curve, d_rates_times, n_rates,
        d_pv_divs_by_tenor_norm,
        d_div_events_by_tenor_norm,
        d_n_div_events_by_tenor_norm,
        time_steps, space_steps,
        n_sigma,
        S_ref
    );

    // ---- 5) Actual IV solve (absolute dividends + absolute option prices)
    {
        constexpr float iv_eps = 0.0f;
        constexpr float guess_width = 0.1f;
        int block_size = 256;
        int grid_size = (n_options + block_size - 1) / block_size;

        compute_fd_iv_kernel_with_guess_tenor_cache<<<grid_size, block_size>>>(
            d_prices, d_spots, d_strikes, d_tenors,
            d_rights, n_options,
            d_tenor_ids,
            d_pv_divs_by_tenor_abs,
            d_div_events_by_tenor_abs,
            d_n_div_events_by_tenor_abs,
            d_rates_curve, d_rates_times, n_rates,
            d_sigma_guess,
            d_results,
            tol, max_iter, v_min, v_max,
            iv_eps,
            guess_width,
            time_steps, space_steps
        );
    }

    cudaDeviceSynchronize();
    cudaMemcpy(results.data(), d_results, n_options * sizeof(float), cudaMemcpyDeviceToHost);

    // ---- 6) Free
    cudaFree(d_sigma_guess);

    cudaFree(d_prices);
    cudaFree(d_spots);
    cudaFree(d_strikes);
    cudaFree(d_tenors);
    cudaFree(d_rights);
    cudaFree(d_results);

    cudaFree(d_tenor_ids);
    cudaFree(d_tenors_unique);

    cudaFree(d_pv_divs_by_tenor_abs);
    cudaFree(d_div_events_by_tenor_abs);
    cudaFree(d_n_div_events_by_tenor_abs);

    cudaFree(d_pv_divs_by_tenor_norm);
    cudaFree(d_div_events_by_tenor_norm);
    cudaFree(d_n_div_events_by_tenor_norm);

    if (d_rates_curve) cudaFree(d_rates_curve);
    if (d_rates_times) cudaFree(d_rates_times);

    return results;
}