
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <vector>
#include <stdexcept>

__device__ inline float get_rate_at_time_old_delete(float t, const float* rates_curve, const float* rates_times, int n_points) {
        if (n_points <= 0) return 0.0f;
        if (n_points == 1 || t <= rates_times[0]) return rates_curve[0];
        if (t >= rates_times[n_points - 1]) return rates_curve[n_points - 1];

        for (int i = 0; i < n_points - 1; ++i) {
            if (t <= rates_times[i + 1]) {
                float ratio = (t - rates_times[i]) / (rates_times[i + 1] - rates_times[i]);
                return rates_curve[i] + ratio * (rates_curve[i + 1] - rates_curve[i]);
            }
        }
        return rates_curve[n_points - 1];
    }

__device__ inline float get_rate_at_time(float t, const float* rates_curve, const float* rates_times, int n_points) {
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

    __device__ float price_american_binomial_cash_div_threadlocal(
        float S, float K, float T,
        float sigma, uint8_t is_call, int n_steps,
        const float* rates_curve, const float* rates_times, int n_rates,
        const float* div_amounts, const int* div_steps, int n_divs)
    {
        constexpr int MAX_STEPS = 512;
        float V[MAX_STEPS + 1];

        // Defensive bound for exponent
        if (n_steps > MAX_STEPS) n_steps = MAX_STEPS;

        // Scaling time steps for short T
        int effective_steps = n_steps;
        if (T < 0.1f) {
            effective_steps = (int)(T * n_steps * 10);
            if (effective_steps > n_steps) effective_steps = n_steps;
            if (effective_steps < 10) effective_steps = 10;
        }
        if (effective_steps > MAX_STEPS) effective_steps = MAX_STEPS;
        float dt = T / effective_steps;

        if (dt <= 0.0f || dt > 0.01f) {
            return is_call ? fmaxf(S - K, 0.0f) : fmaxf(K - S, 0.0f);
        }
        if (dt < 1e-6f) {
            return is_call ? fmaxf(S - K, 0.0f) : fmaxf(K - S, 0.0f);
        }

        float u = expf(sigma * sqrtf(dt));
        if (u <= 1.001f) u = 1.001f;
        float d = 1.0f / u;
        int N = effective_steps;

        // 1. Build the vector of stock prices at maturity
        float u2 = u * u;
        float curr_S = S * powf(d, N);

        for (int j = 0; j <= N; ++j) {
            float S_j = curr_S;
            if (n_divs > 0) {
                for (int div_i = 0; div_i < n_divs; ++div_i) {
                    int step_div = div_steps[div_i];
                    int ups_before_div = (step_div <= j) ? step_div : j;
                    int downs_before_div = step_div - ups_before_div;
                    S_j -= div_amounts[div_i] * powf(u, j - ups_before_div) * powf(d, (N - j) - downs_before_div);
                    if (S_j < 1e-8f) S_j = 1e-8f;
                }
            }
            V[j] = is_call ? fmaxf(S_j - K, 0.0f) : fmaxf(K - S_j, 0.0f);
            curr_S *= u2;
        }

        // 2. Backward induction with term structure
        for (int step = N - 1; step >= 0; --step) {
            // Interpolate rate for the current time step
            float r = get_rate_at_time(step * dt, rates_curve, rates_times, n_rates);
            float disc = expf(-r * dt);
            float edtr = expf(r * dt);
            float p = (edtr - d) / (u - d);
            p = fmaxf(0.0f, fminf(1.0f, p));

            for (int j = 0; j <= step; ++j) {
                float S_j = S * powf(u, j) * powf(d, step - j);
                if (n_divs > 0) {
                    for (int div_i = 0; div_i < n_divs; ++div_i) {
                        int step_div = div_steps[div_i];
                        if (step_div <= step) {
                            int ups_before_div = (step_div <= j) ? step_div : j;
                            int downs_before_div = step_div - ups_before_div;
                            S_j -= div_amounts[div_i] * powf(u, j - ups_before_div) * powf(d, (step - j) - downs_before_div);
                            if (S_j < 1e-8f) S_j = 1e-8f;
                        }
                    }
                }
                float cont = disc * (p * V[j + 1] + (1.0f - p) * V[j]);
                float ex = is_call ? fmaxf(S_j - K, 0.0f) : fmaxf(K - S_j, 0.0f);
                V[j] = fmaxf(cont, ex);
            }
        }
        return V[0];
    }

__device__ float price_american_binomial_cash_div_threadlocal_old_delete_me(
    float S, float K, float T, float r, float sigma, uint8_t is_call, int n_steps,
    const float* div_amounts, const int* div_steps, int n_divs)
{
    constexpr int MAX_STEPS = 512;
    float V[MAX_STEPS + 1];

    // Defensive bound for exponent
    if (n_steps > MAX_STEPS) n_steps = MAX_STEPS;

    // Scaling time steps for short T (copied from original)
    int effective_steps = n_steps;
    if (T < 0.1f) {
        effective_steps = (int)(T * n_steps * 10);
        if (effective_steps > n_steps) effective_steps = n_steps;
        if (effective_steps < 10) effective_steps = 10;
    }
    if (effective_steps > MAX_STEPS) effective_steps = MAX_STEPS;
    float dt = T / effective_steps;

    if (dt <= 0.0f || dt > 0.01f) {
        return is_call ? fmaxf(S - K, 0.0f) : fmaxf(K - S, 0.0f);
    }
    if (dt < 1e-6f) {
        return is_call ? fmaxf(S - K, 0.0f) : fmaxf(K - S, 0.0f);
    }

    float u = expf(sigma * sqrtf(dt));
    if (u <= 1.001f) u = 1.001f;
    if (u <= 1.0f) u = 1.000001f;
    float d = 1.0f / u;
    float disc = expf(-r * dt);
    float edtr = expf(r * dt);
    float p = (edtr - d) / (u - d);
    if (p <= 0.0f || p >= 1.0f || !isfinite(p)) {
        return is_call ? fmaxf(S - K, 0.0f) : fmaxf(K - S, 0.0f);
    }
    p = fmaxf(0.0f, fminf(1.0f, p));

    int N = effective_steps;

    // Precompute u and d multipliers to avoid powf in the loop
    float u2 = u * u;
    float curr_S = S * powf(d, N); // Start at the bottom node (all down moves)

    // 1. Build the vector of stock prices at maturity, with all dividends applied.
    // We'll do this per-path for each end node.
    for (int j = 0; j <= N; ++j) {
        float S_j = curr_S;
        // int n_up = j;
        // int n_down = N - j;
        // Build the actual exercise path: Up N times, Down N-j times in some order.
        // The step at which dividend occurs depends on step
        // Because dividend is at discrete step, we must apply at steps = div_steps[i]
        // For each step, know how many ups happened so far:
        // Since order doesn't matter for stock, we can do the math up front.
        // S_j *= powf(u, n_up) * powf(d, n_down);

        // Rewind: For each dividend, subtract the amount at the step (if any).
        // To accurately do this, "reverse-apply" for all dividends that come before expiry.
        // To mimic the classic "jump-adjusted" binomial, we need to subtract any dividend at its div_steps time,
        // which occur during the tree evolution.
        // For each dividend, adjust S_j back for amount *at the node*:
        if (n_divs > 0) {
            // Walk through all dividends
            for (int div_i = 0; div_i < n_divs; ++div_i) {
                int step_div = div_steps[div_i];
                // The number of up-moves before the dividend is counted from the beginning,
                // so the number of up-moves before step_div is min(j, step_div)
                int ups_before_div = (step_div <= j) ? step_div : j;
                int downs_before_div = step_div - ups_before_div;
                // Calculate the stock price path up to div step
                // float S_before = S * powf(u, ups_before_div) * powf(d, downs_before_div);
                // At this node, subtract the dividend (apply at step_div)
                S_j -= div_amounts[div_i] * powf(u, j - ups_before_div) * powf(d, (N - j) - downs_before_div);
                // Stock can not become negative
                if (S_j < 1e-8f) S_j = 1e-8f;
            }
        }
        V[j] = is_call ? fmaxf(S_j - K, 0.0f) : fmaxf(K - S_j, 0.0f);
        curr_S *= u2; // Move to the next node: replaces one 'd' with one 'u'
    }

    // 2. Backward induction
    // For American options, need early exercise check at each node during backward step.
    for (int step = N - 1; step >= 0; --step) {
        for (int j = 0; j <= step; ++j) {
            // At (step, j), path is: j up moves, (step - j) down moves
            // Rebuild stock price at (step, j) from S, dividends
            float S_j = S * powf(u, j) * powf(d, step - j);
            if (n_divs > 0) {
                for (int div_i = 0; div_i < n_divs; ++div_i) {
                    int step_div = div_steps[div_i];
                    if (step_div <= step) {
                        // Dividend paid at step_div
                        int ups_before_div = (step_div <= j) ? step_div : j;
                        int downs_before_div = step_div - ups_before_div;
                        // float S_before = S * powf(u, ups_before_div) * powf(d, downs_before_div);
                        S_j -= div_amounts[div_i] * powf(u, j - ups_before_div) * powf(d, (step - j) - downs_before_div);
                        if (S_j < 1e-8f) S_j = 1e-8f;
                    }
                }
            }
            float cont = disc * (p * V[j + 1] + (1.0f - p) * V[j]);
            float ex = is_call ? fmaxf(S_j - K, 0.0f) : fmaxf(K - S_j, 0.0f);
            V[j] = fmaxf(cont, ex);
        }
    }
    return V[0];
}

__device__ float implied_vol_american_bisection_threadlocal(
    float target, float S, float K, float T, uint8_t is_call, int n_steps,
    const float* rates_curve, const float* rates_times, int n_rates,
    const float* div_amounts, const int* div_steps, int n_divs,
    float tol = 1e-6f, int max_iter = 100, float v_min = 1e-4f, float v_max = 5.0f)
{
    float p_low = price_american_binomial_cash_div_threadlocal(S, K, T,  v_min, is_call, n_steps, rates_curve, rates_times, n_rates, div_amounts, div_steps, n_divs);
    float p_high = price_american_binomial_cash_div_threadlocal(S, K, T, v_max, is_call, n_steps, rates_curve, rates_times, n_rates, div_amounts, div_steps, n_divs);
    if (target <= p_low + 1e-12f) return 0.0f;
    if (target > p_high + 1e-12f) return NAN;

    float lo = v_min, hi = v_max, mid = 0.0f;
    for (int i = 0; i < max_iter; ++i) {
        mid = 0.5f * (lo + hi);
        float p_mid = price_american_binomial_cash_div_threadlocal(S, K, T, mid, is_call, n_steps, rates_curve, rates_times, n_rates, div_amounts, div_steps, n_divs);
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
    const float* rates_curve, const float* rates_times, int n_rates,
    const float* div_amounts, const int* div_steps, int n_divs,
    float rel_shift = 1e-4f)
{
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
    const float* rates_curve, const float* rates_times, int n_rates,
    const float* div_amounts, const int* div_steps, int n_divs,
    float abs_shift = 1e-4f)
{
    float h = fmaxf(abs_shift, sigma * 1e-2f);  // Robust shift for high vols
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
    const float* prices, const float* spots, const float* strikes, const float* tenors,
    const uint8_t* v_is_call, const float* rates_curve, const float* rates_times, int n_rates,
    int n_options,
    const float* div_amounts, const float* div_times, int n_divs,
    float* results,
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
            int step = (int)roundf(div_time / t * n_steps);
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

std::vector<float> get_v_iv_fd_cuda(
    const std::vector<float>& prices,
    const std::vector<float>& spots,
    const std::vector<float>& strikes,
    const std::vector<float>& tenors,
    const std::vector<uint8_t>& v_is_call,
    const std::vector<float>& rates_curve = {},
    const std::vector<float>& rates_times = {},
    const std::vector<float>& div_amounts = {},
    const std::vector<float>& div_times = {},
    const float tol = 1e-6f,
    const int max_iter = 200,
    const float v_min = 1e-7f,
    const float v_max = 5.0f,
    const float steps_factor = 1
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

float price_american_binomial_cpu(
    float S, float K, float T, float sigma, uint8_t is_call, int n_steps,
    const std::vector<float>& rates_curve, const std::vector<float>& rates_times,
    const std::vector<float>& div_amounts, const std::vector<float>& div_times)
{
    if (T <= 0) return is_call ? std::max(S - K, 0.0f) : std::max(K - S, 0.0f);

    // 1. PRE-CALCULATE CONSTANTS
    const float dt = T / n_steps;
    const float v_sq = sigma * sigma;
    const int n_rates = rates_curve.size();

    // Helper for CPU interpolation
    auto get_rate_cpu = [&](float t) {
        if (n_rates <= 0) return 0.0f;
        if (t <= rates_times[0]) return rates_curve[0];
        if (t >= rates_times[n_rates - 1]) {
            // Beyond the last point, we usually assume the last instantaneous forward rate continues
            // For simplicity, return the last zero rate, but ideally calculate the last forward.
            return rates_curve[n_rates - 1];
        }

        // Find the interval [i, i+1]
        for (int i = 0; i < n_rates - 1; ++i) {
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
        return rates_curve[n_rates - 1];
    };

        // 2. STACK ALLOCATION
        float V[513];
        int N = (n_steps > 512) ? 512 : n_steps;

        // 3. INITIALIZE TERMINAL NODES
        // For JR with term structure, we need the total drift up to T
        float total_drift = 0;
        for (int i = 0; i < N; ++i) {
            total_drift += (get_rate_cpu(i * dt) - 0.5f * v_sq) * dt;
        }

        const float sigma_sqrt_dt = sigma * std::sqrt(dt);
        const float u_base = std::exp(sigma_sqrt_dt);
        const float d_base = std::exp(-sigma_sqrt_dt);
        const float drift_total_exp = std::exp(total_drift);

        for (int j = 0; j <= N; ++j) {
            // S_T = S * exp(sum(r_i - 0.5*sig^2)dt) * u_base^j * d_base^(N-j)
            float St = S * drift_total_exp * std::pow(u_base, j) * std::pow(d_base, N - j);

            if (!div_amounts.empty()) {
                for (size_t i = 0; i < div_amounts.size(); ++i) {
                    if (div_times[i] < T) St = std::max(St - div_amounts[i], 0.0f);
                }
            }
            V[j] = is_call ? std::max(St - K, 0.0f) : std::max(K - St, 0.0f);
        }

        // 4. BACKWARD INDUCTION
        const float p = 0.5f; // Jarrow-Rudd fixed probability
        for (int i = N - 1; i >= 0; --i) {
            float r = get_rate_cpu(i * dt);
            float disc = std::exp(-r * dt);
            float p_disc = p * disc;
            float q_disc = (1.0f - p) * disc;

            // To check early exercise, we need S at this specific node (i, j)
            // Cumulative drift up to time i*dt
            float drift_to_step = 0;
            for(int k=0; k<i; ++k) drift_to_step += (get_rate_cpu(k*dt) - 0.5f * v_sq) * dt;
            float drift_exp = std::exp(drift_to_step);

            for (int j = 0; j <= i; ++j) {
                float continuation = p_disc * V[j + 1] + q_disc * V[j];

                float S_node = S * drift_exp * std::pow(u_base, j) * std::pow(d_base, i - j);
                float exercise = is_call ? std::max(S_node - K, 0.0f) : std::max(K - S_node, 0.0f);

                V[j] = std::max(exercise, continuation);
            }
        }
        return V[0];
}

// float price_american_binomial_cpu_old_delete(
//     float S, float K, float T, float sigma, uint8_t is_call, int n_steps,
//     const std::vector<float>& rates_curve, const std::vector<float>& rates_times,
//     const std::vector<float>& div_amounts, const std::vector<float>& div_times)
// {
//     if (T <= 0) return is_call ? std::max(S - K, 0.0f) : std::max(K - S, 0.0f);
//
//     // 1. PRE-CALCULATE CONSTANTS (Outside the loops)
//     const float dt = T / n_steps;
//     const float v_sq = sigma * sigma;
//
//     // Precompute rates
//     // float r_t[MAX_TIME];
//     // for (int step = 0; step < n_steps; ++step) {
//     //     float t = (step + 1) * dt;
//     //     if (n_curve_points <= 1) {
//     //         r_t[step] = (n_curve_points == 1) ? rates_curve[0] : 0.0f;
//     //     } else {
//     //         if (t <= rates_times[0]) {
//     //             r_t[step] = rates_curve[0];
//     //         } else if (t >= rates_times[n_curve_points - 1]) {
//     //             r_t[step] = rates_curve[n_curve_points - 1];
//     //         } else {
//     //             int i = 0;
//     //             for (; i < n_curve_points - 1; ++i) {
//     //                 if (t <= rates_times[i + 1]) break;
//     //             }
//     //             float w = (t - rates_times[i]) / (rates_times[i + 1] - rates_times[i]);
//     //             r_t[step] = rates_curve[i] * (1.0f - w) + rates_curve[i + 1] * w;
//     //         }
//     //     }
//     // }
//
//     // Using the Jarrow-Rudd (Drift-Centered) parameters that matched your QL values
//     const float u = std::exp((r - 0.5f * v_sq) * dt + sigma * std::sqrt(dt));
//     const float d = std::exp((r - 0.5f * v_sq) * dt - sigma * std::sqrt(dt));
//     const float p = 0.5f;
//     const float disc = std::exp(-r * dt);
//     const float p_disc = p * disc;
//     const float q_disc = (1.0f - p) * disc;
//
//     // 2. STACK ALLOCATION (Much faster than std::vector)
//     // Use a fixed size or a small buffer. 512 is plenty for IV.
//     float V[513];
//     int N = (n_steps > 512) ? 512 : n_steps;
//
//     // 3. INITIALIZE TERMINAL NODES (Incremental multiplication instead of std::pow)
//     float curr_S = S * std::pow(d, N);
//     const float u_over_d = u / d;
//
//     for (int j = 0; j <= N; ++j) {
//         float St = curr_S;
//         // Dividend adjustment (only if needed)
//         if (!div_amounts.empty()) {
//             for (size_t i = 0; i < div_amounts.size(); ++i) {
//                 if (div_times[i] < T) St = std::max(St - div_amounts[i], 0.0f);
//             }
//         }
//         V[j] = is_call ? std::max(St - K, 0.0f) : std::max(K - St, 0.0f);
//         curr_S *= u_over_d; // Incremental: S * d^(N-j) * u^j
//     }
//
//     // 4. BACKWARD INDUCTION (Optimized loop)
//     for (int i = N - 1; i >= 0; --i) {
//         // Pre-calculate S for the bottom of this step
//         float S_node = S * std::pow(d, i);
//
//         for (int j = 0; j <= i; ++j) {
//             // Continuation value (Avoid repeated multiplication)
//             float continuation = p_disc * V[j + 1] + q_disc * V[j];
//
//             // Exercise value
//             float exercise = is_call ? std::max(S_node - K, 0.0f) : std::max(K - S_node, 0.0f);
//
//             V[j] = std::max(exercise, continuation);
//             S_node *= u_over_d; // Incremental S for the next node up
//         }
//     }
//     return V[0];
// }

float get_single_iv_cpu(
    float target_price, float S, float K, float T, uint8_t is_call,
    const std::vector<float>& rates_curve, const std::vector<float>& rates_times,
    const std::vector<float>& div_amounts, const std::vector<float>& div_times,
    float tol, int max_iter, const float steps_factor = 1)
{
    float low = 0.0001f, high = 5.0f;

    // We start with a baseline step count
    float current_steps = (T < 0.01 ? 256 : 32) * steps_factor;  // About to expire, need high time resolution from the start.

    // Check boundaries
    if (target_price <= 0) return 0.0f;

    for (int i = 0; i < max_iter; ++i) {
        float mid = (low + high) * 0.5f;
        float price = price_american_binomial_cpu(S, K, T, mid, is_call, static_cast<int>(current_steps), rates_curve, rates_times, div_amounts, div_times);

        if (std::abs(price - target_price) < tol) return mid;
        if (price < target_price) low = mid;
        else high = mid;

        float error = std::abs(price - target_price);
        if (error < tol) break;

        // DECISION LOGIC: Increase steps based on how close we are (error-based)
        // If error is less than 50 cents, use 64 steps
        // If error is less than 5 cents, use 128 steps
        // If error is less than 1 cent, use 256 steps
        if (error < 0.01f)      current_steps = max(current_steps, 256 * steps_factor);
        else if (error < 0.05f) current_steps = max(current_steps, 128 * steps_factor);
        else if (error < 0.50f) current_steps = max(current_steps, 64 * steps_factor);
        else                    current_steps = max(current_steps, 32 * steps_factor);
    }
    return (low + high) * 0.5f;
}

// Device kernels for vectorized price, delta, vega (each option can have distinct sigma)

__global__ void compute_price_kernel_threadlocal(
    const float* spots, const float* strikes, const float* tenors, const float* sigmas,
    const int* rights, int n_steps, int n_options,
    const float* rates_curve, const float* rates_times, int n_rates,
    const float* div_amounts, const float* div_times, int n_divs,
    float* out_price)
{
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

    const int MAX_DIVS = 32;
    float local_div_amounts[MAX_DIVS];
    int local_div_steps[MAX_DIVS];
    int valid_divs = 0;
    for (int j = 0; j < n_divs && j < MAX_DIVS; ++j) {
        float div_time = div_times[j];
        if (div_time > 0.0f && div_time < t) {
            int step = (int)roundf(div_time / t * n_steps);
            step = max(1, min(step, n_steps));
            local_div_steps[valid_divs] = step;
            local_div_amounts[valid_divs] = div_amounts[j];
            valid_divs++;
        }
    }

    out_price[idx] = price_american_binomial_cash_div_threadlocal(
        s, k, t, sigma, right, n_steps,
        rates_curve, rates_times, n_rates,
        local_div_amounts, local_div_steps, valid_divs
    );
}

__global__ void compute_delta_kernel_threadlocal(
    const float* spots, const float* strikes, const float* tenors, const float* sigmas,
    const int* rights, int n_steps, int n_options,
    const float* rates_curve, const float* rates_times, int n_rates,
    const float* div_amounts, const float* div_times, int n_divs,
    float* out_delta)
{
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
            int step = (int)roundf(div_time / t * n_steps);
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
    const float* spots, const float* strikes, const float* tenors, const float* sigmas,
    const uint8_t* v_is_call, int n_steps, int n_options,
    const float* rates_curve, const float* rates_times, int n_rates,
    const float* div_amounts, const float* div_times, int n_divs,
    float* out_vega)
{
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
            int step = (int)roundf(div_time / t * n_steps);
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

// Host wrappers: Each returns vector<float>

std::vector<float> get_v_fd_price_cuda(
    const std::vector<float>& spots,
    const std::vector<float>& strikes,
    const std::vector<float>& tenors,
    const std::vector<uint8_t>& v_is_call,
    const std::vector<float>& sigmas,
    int n_steps = 100,
    const std::vector<float>& rates_curve = {},
    const std::vector<float>& rates_times = {},
    const std::vector<float>& div_amounts = {},
    const std::vector<float>& div_times = {}
    )
{
    int n_options = spots.size();
    std::vector<float> prices(n_options);

    if (n_options == 0) return prices;

    int n_divs = div_amounts.size();
    int n_rates = rates_curve.size();

    float *d_spots, *d_strikes, *d_tenors, *d_sigmas;
    int *d_v_is_call;
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
        cudaMalloc(&d_div_times, n_rates * sizeof(float));
        cudaMemcpy(d_rates_curve, rates_curve.data(), n_rates * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_rates_times, rates_times.data(), n_rates * sizeof(float), cudaMemcpyHostToDevice);
    }

    int block_size = 256;
    int grid_size = (n_options + block_size - 1) / block_size;

    compute_price_kernel_threadlocal<<<grid_size, block_size>>>(
        d_spots, d_strikes, d_tenors, d_sigmas, d_v_is_call, n_steps, n_options,
        d_rates_curve, d_rates_times, n_rates,
        d_div_amounts, d_div_times, n_divs, d_prices
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
    const std::vector<float>& spots,
    const std::vector<float>& strikes,
    const std::vector<float>& tenors,
    const std::vector<uint8_t>& v_is_call,
    const std::vector<float>& sigmas,
    int n_steps = 100,
    const std::vector<float>& rates_curve = {},
    const std::vector<float>& rates_times = {},
    const std::vector<float>& div_amounts = {},
    const std::vector<float>& div_times = {})
{
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
        cudaMalloc(&d_div_times, n_rates * sizeof(float));
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
    const std::vector<float>& spots,
    const std::vector<float>& strikes,
    const std::vector<float>& tenors,
    const std::vector<uint8_t>& v_is_call,
    const std::vector<float>& sigmas,
    int n_steps = 100,
    const std::vector<float>& rates_curve = {},
    const std::vector<float>& rates_times = {},
    const std::vector<float>& div_amounts = {},
    const std::vector<float>& div_times = {})
{
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
        cudaMalloc(&d_div_times, n_rates * sizeof(float));
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
