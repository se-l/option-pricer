
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <vector>
#include <stdexcept>

// =======================
// Major Fix: All tree/temp arrays are per-thread, local stack arrays, not shared_mem
// =======================

__device__ float price_american_binomial_cash_div_threadlocal(
    float S, float K, float T, float r, float sigma, bool is_call, int n_steps,
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
    float target, float S, float K, float T, float r, int is_call, int n_steps,
    const float* div_amounts, const int* div_steps, int n_divs,
    float tol = 1e-6f, int max_iter = 100, float v_min = 1e-4f, float v_max = 5.0f)
{
    float p_low = price_american_binomial_cash_div_threadlocal(S, K, T, r, v_min, is_call, n_steps, div_amounts, div_steps, n_divs);
    float p_high = price_american_binomial_cash_div_threadlocal(S, K, T, r, v_max, is_call, n_steps, div_amounts, div_steps, n_divs);
    if (target <= p_low + 1e-12f) return 0.0f;
    if (target > p_high + 1e-12f) return NAN;

    float lo = v_min, hi = v_max, mid = 0.0f;
    for (int i = 0; i < max_iter; ++i) {
        mid = 0.5f * (lo + hi);
        float p_mid = price_american_binomial_cash_div_threadlocal(S, K, T, r, mid, is_call, n_steps, div_amounts, div_steps, n_divs);
        float diff = p_mid - target;
        if (fabsf(diff) < tol) return mid;
        if (diff < 0.0f) lo = mid;
        else hi = mid;
        if (hi - lo < tol) break;
    }
    return mid;
}

__device__ float american_binomial_delta_fd(
    float S, float K, float T, float r, float sigma, int is_call, int n_steps,
    const float* div_amounts, const int* div_steps, int n_divs,
    float rel_shift = 1e-4f)
{
    float h = S * rel_shift;
    if (h < 1e-6f) h = 1e-6f;
    float up = price_american_binomial_cash_div_threadlocal(
        S + h, K, T, r, sigma, is_call, n_steps, div_amounts, div_steps, n_divs);
    float down = price_american_binomial_cash_div_threadlocal(
        S - h, K, T, r, sigma, is_call, n_steps, div_amounts, div_steps, n_divs);
    return (up - down) / (2.0f * h);
}

__device__ float american_binomial_vega_fd(
    float S, float K, float T, float r, float sigma, int is_call, int n_steps,
    const float* div_amounts, const int* div_steps, int n_divs,
    float abs_shift = 1e-4f)
{
    float h = fmaxf(abs_shift, sigma * 1e-2f);  // Robust shift for high vols
    float up = price_american_binomial_cash_div_threadlocal(
        S, K, T, r, sigma + h, is_call, n_steps, div_amounts, div_steps, n_divs);
    float down = price_american_binomial_cash_div_threadlocal(
        S, K, T, r, sigma - h, is_call, n_steps, div_amounts, div_steps, n_divs);
    return (up - down) / (2.0f * h);
}

__device__ float implied_vol_american_newton_threadlocal(
        float target, float S, float K, float T, float r, int is_call, int n_steps,
        const float* div_amounts, const int* div_steps, int n_divs,
        float tol = 1e-5f, int max_iter = 20, float v_min = 1e-4f, float v_max = 5.0f,
        float initial_guess = 0.3f)
{
    // Initial guess: Corrado-Miller or simple 0.3
    float sigma = initial_guess;

    for (int i = 0; i < max_iter; ++i) {
        float p = price_american_binomial_cash_div_threadlocal(S, K, T, r, sigma, is_call, n_steps, div_amounts, div_steps, n_divs);
        float diff = p - target;
        if (fabsf(diff) < tol) return sigma;

        float v = american_binomial_vega_fd(S, K, T, r, sigma, is_call, n_steps, div_amounts, div_steps, n_divs);

        // Avoid division by zero or tiny vega
        if (fabsf(v) < 1e-6f) {
            // Fallback to a small bisection step if vega is too small
            sigma += (diff > 0) ? -0.01f : 0.01f;
        } else {
            float step = diff / v;
            sigma -= step;
        }

        if (sigma <= 0.0f) sigma = v_min;
        if (sigma > v_max) sigma = v_max;
    }
    return sigma;
}

// =======================
// Main IV kernel: no shared memory needed
// =======================

__global__ void compute_iv_kernel_threadlocal(
    const float* prices, const float* spots, const float* strikes, const float* tenors,
    const int* rights, float r, int n_steps, int n_options,
    const float* div_amounts, const float* div_times, int n_divs,
    float* results,
    float tol = 1e-6f, int max_iter = 100, float v_min = 1e-4f, float v_max = 5.0f) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_options) return;

    float p = prices[idx];
    float s = spots[idx];
    float k = strikes[idx];
    float t = tenors[idx];
    int right = rights[idx];

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

    float intrinsic = (right == 1) ? max(s - k, 0.0f) : max(k - s, 0.0f);
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
        p, s, k, t, r, right, n_steps,
        local_div_amounts, local_div_steps, valid_divs,
        tol, max_iter, v_min, v_max
    );
}

// Host wrapper using new kernel (Python binding unchanged, just use this implementation)

std::vector<float> get_v_iv_fd_cuda(
    const std::vector<float>& prices,
    const std::vector<float>& spots,
    const std::vector<float>& strikes,
    const std::vector<float>& tenors,
    const std::vector<std::string>& rights,
    float r,
    int n_steps = 200,
    const std::vector<float>& div_amounts = {},
    const std::vector<float>& div_times = {},
    float tol = 1e-6f,
    int max_iter = 100,
    float v_min = 1e-4f,
    float v_max = 5.0f
    ) {

    int n_options = prices.size();
    std::vector<float> results(n_options);

    if (n_options == 0) return results;

    // Convert rights to int
    std::vector<int> rights_int(n_options);
    for (int i = 0; i < n_options; ++i) {
        rights_int[i] = (rights[i] == "c" || rights[i] == "C" ||
                        rights[i] == "call" || rights[i] == "CALL") ? 1 : 0;
    }

    int n_divs = div_amounts.size();

    // Allocate device memory (inputs + outputs)
    float *d_prices, *d_spots, *d_strikes, *d_tenors, *d_results;
    int *d_rights;
    float *d_div_amounts = nullptr, *d_div_times = nullptr;

    cudaMalloc(&d_prices, n_options * sizeof(float));
    cudaMalloc(&d_spots, n_options * sizeof(float));
    cudaMalloc(&d_strikes, n_options * sizeof(float));
    cudaMalloc(&d_tenors, n_options * sizeof(float));
    cudaMalloc(&d_rights, n_options * sizeof(int));
    cudaMalloc(&d_results, n_options * sizeof(float));

    cudaMemcpy(d_prices, prices.data(), n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_spots, spots.data(), n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_strikes, strikes.data(), n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tenors, tenors.data(), n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rights, rights_int.data(), n_options * sizeof(int), cudaMemcpyHostToDevice);

    if (n_divs > 0 && div_amounts.size() == div_times.size()) {
        cudaMalloc(&d_div_amounts, n_divs * sizeof(float));
        cudaMalloc(&d_div_times, n_divs * sizeof(float));
        cudaMemcpy(d_div_amounts, div_amounts.data(), n_divs * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_div_times, div_times.data(), n_divs * sizeof(float), cudaMemcpyHostToDevice);
    }

    // Launch with no shared_mem needed
    int block_size = 256;
    int grid_size = (n_options + block_size - 1) / block_size;

    compute_iv_kernel_threadlocal<<<grid_size, block_size>>>(
        d_prices, d_spots, d_strikes, d_tenors, d_rights, r, n_steps, n_options,
        d_div_amounts, d_div_times, n_divs, d_results,
        tol, max_iter, v_min, v_max
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));

    cudaMemcpy(results.data(), d_results, n_options * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_prices);
    cudaFree(d_spots);
    cudaFree(d_strikes);
    cudaFree(d_tenors);
    cudaFree(d_rights);
    cudaFree(d_results);
    if (d_div_amounts) cudaFree(d_div_amounts);
    if (d_div_times) cudaFree(d_div_times);

    return results;
}

// ... existing code ...
__global__ void compute_single_iv_parallel_kernel(
    float target, float S, float K, float T, float r, bool is_call, int n_steps,
    const float* d_div_amounts, const int* d_div_steps, int n_divs,
    float v_min, float v_max, float* d_out_bracket)
{
    // Each thread calculates the price for a specific volatility in the current bracket
    int idx = threadIdx.x;
    int num_threads = blockDim.x;

    float step = (v_max - v_min) / (num_threads - 1);
    float sigma = v_min + idx * step;

    float price = price_american_binomial_cash_div_threadlocal(
        S, K, T, r, sigma, is_call, n_steps, d_div_amounts, d_div_steps, n_divs
    );

    // We store the difference to the target price
    d_out_bracket[idx] = price - target;
}

float get_single_iv_cuda(
    float price, float S, float K, float T, const bool is_call, float r,
    int n_steps = 200,
    const std::vector<float>& div_amounts = {},
    const std::vector<float>& div_times = {},
    float tol = 1e-5f, int max_outer_iters = 5)
{
    // 1. Setup Dividends
    int n_divs = div_amounts.size();
    std::vector<int> div_steps(n_divs);
    for(int i=0; i<n_divs; ++i) {
        div_steps[i] = static_cast<int>(roundf(div_times[i] / T * n_steps));
    }

    float *d_div_amounts = nullptr, *d_results;
    int *d_div_steps = nullptr;

    const int num_samples = 256; // Use 256 threads to split the bracket
    cudaMalloc(&d_results, num_samples * sizeof(float));
    if (n_divs > 0) {
        cudaMalloc(&d_div_amounts, n_divs * sizeof(float));
        cudaMalloc(&d_div_steps, n_divs * sizeof(int));
        cudaMemcpy(d_div_amounts, div_amounts.data(), n_divs * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_div_steps, div_steps.data(), n_divs * sizeof(int), cudaMemcpyHostToDevice);
    }

    float v_min = 1e-4f, v_max = 4.0f;
    std::vector<float> h_results(num_samples);

    // 2. Parallel Bracketing Loop
    // Each iteration reduces the search space by a factor of 255
    for (int iter = 0; iter < max_outer_iters; ++iter) {
        compute_single_iv_parallel_kernel<<<1, num_samples>>>(
            price, S, K, T, r, is_call, n_steps, d_div_amounts, d_div_steps, n_divs,
            v_min, v_max, d_results
        );
        cudaMemcpy(h_results.data(), d_results, num_samples * sizeof(float), cudaMemcpyDeviceToHost);

        // Find where the sign flips (where price - target crossing zero)
        int cross_idx = 0;
        for (int i = 0; i < num_samples - 1; ++i) {
            if (h_results[i] * h_results[i+1] <= 0) {
                cross_idx = i;
                break;
            }
        }

        float new_v_min = v_min + cross_idx * (v_max - v_min) / (num_samples - 1);
        float new_v_max = v_min + (cross_idx + 1) * (v_max - v_min) / (num_samples - 1);

        v_min = new_v_min;
        v_max = new_v_max;

        if ((v_max - v_min) < tol) break;
    }

    // Cleanup
    cudaFree(d_results);
    if (d_div_amounts) { cudaFree(d_div_amounts); cudaFree(d_div_steps); }

    return (v_min + v_max) * 0.5f;
}

float price_american_binomial_cpu(
    float S, float K, float T, float r, float sigma, bool is_call, int n_steps,
    const std::vector<float>& div_amounts, const std::vector<float>& div_times)
{
    if (T <= 0) return is_call ? std::max(S - K, 0.0f) : std::max(K - S, 0.0f);

    // 1. PRE-CALCULATE CONSTANTS (Outside the loops)
    const float dt = T / n_steps;
    const float v_sq = sigma * sigma;

    // Using the Jarrow-Rudd (Drift-Centered) parameters that matched your QL values
    const float u = std::exp((r - 0.5f * v_sq) * dt + sigma * std::sqrt(dt));
    const float d = std::exp((r - 0.5f * v_sq) * dt - sigma * std::sqrt(dt));
    const float p = 0.5f;
    const float disc = std::exp(-r * dt);
    const float p_disc = p * disc;
    const float q_disc = (1.0f - p) * disc;

    // 2. STACK ALLOCATION (Much faster than std::vector)
    // Use a fixed size or a small buffer. 512 is plenty for IV.
    float V[513];
    int N = (n_steps > 512) ? 512 : n_steps;

    // 3. INITIALIZE TERMINAL NODES (Incremental multiplication instead of std::pow)
    float curr_S = S * std::pow(d, N);
    const float u_over_d = u / d;

    for (int j = 0; j <= N; ++j) {
        float St = curr_S;
        // Dividend adjustment (only if needed)
        if (!div_amounts.empty()) {
            for (size_t i = 0; i < div_amounts.size(); ++i) {
                if (div_times[i] < T) St = std::max(St - div_amounts[i], 0.0f);
            }
        }
        V[j] = is_call ? std::max(St - K, 0.0f) : std::max(K - St, 0.0f);
        curr_S *= u_over_d; // Incremental: S * d^(N-j) * u^j
    }

    // 4. BACKWARD INDUCTION (Optimized loop)
    for (int i = N - 1; i >= 0; --i) {
        // Pre-calculate S for the bottom of this step
        float S_node = S * std::pow(d, i);

        for (int j = 0; j <= i; ++j) {
            // Continuation value (Avoid repeated multiplication)
            float continuation = p_disc * V[j + 1] + q_disc * V[j];

            // Exercise value
            float exercise = is_call ? std::max(S_node - K, 0.0f) : std::max(K - S_node, 0.0f);

            V[j] = std::max(exercise, continuation);
            S_node *= u_over_d; // Incremental S for the next node up
        }
    }
    return V[0];
}

float get_single_iv_cpu(
    float target_price, float S, float K, float T, bool is_call, float r,
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
        float price = price_american_binomial_cpu(S, K, T, r, mid, is_call, static_cast<int>(current_steps), div_amounts, div_times);

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
    const int* rights, float r, int n_steps, int n_options,
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
        s, k, t, r, sigma, right, n_steps,
        local_div_amounts, local_div_steps, valid_divs
    );
}

__global__ void compute_delta_kernel_threadlocal(
    const float* spots, const float* strikes, const float* tenors, const float* sigmas,
    const int* rights, float r, int n_steps, int n_options,
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
        s, k, t, r, sigma, right, n_steps,
        local_div_amounts, local_div_steps, valid_divs
    );
}

__global__ void compute_vega_kernel_threadlocal(
    const float* spots, const float* strikes, const float* tenors, const float* sigmas,
    const int* rights, float r, int n_steps, int n_options,
    const float* div_amounts, const float* div_times, int n_divs,
    float* out_vega)
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
        s, k, t, r, sigma, right, n_steps,
        local_div_amounts, local_div_steps, valid_divs
    );
}

// Host wrappers: Each returns vector<float>

std::vector<float> get_v_fd_price_cuda(
    const std::vector<float>& spots,
    const std::vector<float>& strikes,
    const std::vector<float>& tenors,
    const std::vector<std::string>& rights,
    const std::vector<float>& sigmas,
    float r,
    int n_steps = 100,
    const std::vector<float>& div_amounts = {},
    const std::vector<float>& div_times = {})
{
    int n_options = spots.size();
    std::vector<float> prices(n_options);

    if (n_options == 0) return prices;

    std::vector<int> rights_int(n_options);
    for (int i = 0; i < n_options; ++i) {
        rights_int[i] = (rights[i] == "c" || rights[i] == "C" ||
                        rights[i] == "call" || rights[i] == "CALL") ? 1 : 0;
    }

    int n_divs = div_amounts.size();

    float *d_spots, *d_strikes, *d_tenors, *d_sigmas;
    int *d_rights;
    float *d_div_amounts = nullptr, *d_div_times = nullptr;
    float *d_prices;

    cudaMalloc(&d_spots, n_options * sizeof(float));
    cudaMalloc(&d_strikes, n_options * sizeof(float));
    cudaMalloc(&d_tenors, n_options * sizeof(float));
    cudaMalloc(&d_sigmas, n_options * sizeof(float));
    cudaMalloc(&d_rights, n_options * sizeof(int));
    cudaMalloc(&d_prices, n_options * sizeof(float));

    cudaMemcpy(d_spots, spots.data(), n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_strikes, strikes.data(), n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tenors, tenors.data(), n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sigmas, sigmas.data(), n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rights, rights_int.data(), n_options * sizeof(int), cudaMemcpyHostToDevice);

    if (n_divs > 0 && div_amounts.size() == div_times.size()) {
        cudaMalloc(&d_div_amounts, n_divs * sizeof(float));
        cudaMalloc(&d_div_times, n_divs * sizeof(float));
        cudaMemcpy(d_div_amounts, div_amounts.data(), n_divs * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_div_times, div_times.data(), n_divs * sizeof(float), cudaMemcpyHostToDevice);
    }

    int block_size = 256;
    int grid_size = (n_options + block_size - 1) / block_size;

    compute_price_kernel_threadlocal<<<grid_size, block_size>>>(
        d_spots, d_strikes, d_tenors, d_sigmas, d_rights, r, n_steps, n_options,
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
    cudaFree(d_rights);
    cudaFree(d_prices);
    if (d_div_amounts) cudaFree(d_div_amounts);
    if (d_div_times) cudaFree(d_div_times);

    return prices;
}

std::vector<float> get_v_fd_delta_cuda(
    const std::vector<float>& spots,
    const std::vector<float>& strikes,
    const std::vector<float>& tenors,
    const std::vector<std::string>& rights,
    const std::vector<float>& sigmas,
    float r,
    int n_steps = 100,
    const std::vector<float>& div_amounts = {},
    const std::vector<float>& div_times = {})
{
    int n_options = spots.size();
    std::vector<float> deltas(n_options);

    if (n_options == 0) return deltas;

    std::vector<int> rights_int(n_options);
    for (int i = 0; i < n_options; ++i) {
        rights_int[i] = (rights[i] == "c" || rights[i] == "C" ||
                        rights[i] == "call" || rights[i] == "CALL") ? 1 : 0;
    }

    int n_divs = div_amounts.size();

    float *d_spots, *d_strikes, *d_tenors, *d_sigmas;
    int *d_rights;
    float *d_div_amounts = nullptr, *d_div_times = nullptr;
    float *d_deltas;

    cudaMalloc(&d_spots, n_options * sizeof(float));
    cudaMalloc(&d_strikes, n_options * sizeof(float));
    cudaMalloc(&d_tenors, n_options * sizeof(float));
    cudaMalloc(&d_sigmas, n_options * sizeof(float));
    cudaMalloc(&d_rights, n_options * sizeof(int));
    cudaMalloc(&d_deltas, n_options * sizeof(float));

    cudaMemcpy(d_spots, spots.data(), n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_strikes, strikes.data(), n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tenors, tenors.data(), n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sigmas, sigmas.data(), n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rights, rights_int.data(), n_options * sizeof(int), cudaMemcpyHostToDevice);

    if (n_divs > 0 && div_amounts.size() == div_times.size()) {
        cudaMalloc(&d_div_amounts, n_divs * sizeof(float));
        cudaMalloc(&d_div_times, n_divs * sizeof(float));
        cudaMemcpy(d_div_amounts, div_amounts.data(), n_divs * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_div_times, div_times.data(), n_divs * sizeof(float), cudaMemcpyHostToDevice);
    }

    int block_size = 256;
    int grid_size = (n_options + block_size - 1) / block_size;

    compute_delta_kernel_threadlocal<<<grid_size, block_size>>>(
        d_spots, d_strikes, d_tenors, d_sigmas, d_rights, r, n_steps, n_options,
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
    cudaFree(d_rights);
    cudaFree(d_deltas);
    if (d_div_amounts) cudaFree(d_div_amounts);
    if (d_div_times) cudaFree(d_div_times);

    return deltas;
}

std::vector<float> get_v_fd_vega_cuda(
    const std::vector<float>& spots,
    const std::vector<float>& strikes,
    const std::vector<float>& tenors,
    const std::vector<std::string>& rights,
    const std::vector<float>& sigmas,
    float r,
    int n_steps = 100,
    const std::vector<float>& div_amounts = {},
    const std::vector<float>& div_times = {})
{
    int n_options = spots.size();
    std::vector<float> vegas(n_options);

    if (n_options == 0) return vegas;

    std::vector<int> rights_int(n_options);
    for (int i = 0; i < n_options; ++i) {
        rights_int[i] = (rights[i] == "c" || rights[i] == "C" ||
                        rights[i] == "call" || rights[i] == "CALL") ? 1 : 0;
    }

    int n_divs = div_amounts.size();

    float *d_spots, *d_strikes, *d_tenors, *d_sigmas;
    int *d_rights;
    float *d_div_amounts = nullptr, *d_div_times = nullptr;
    float *d_vegas;

    cudaMalloc(&d_spots, n_options * sizeof(float));
    cudaMalloc(&d_strikes, n_options * sizeof(float));
    cudaMalloc(&d_tenors, n_options * sizeof(float));
    cudaMalloc(&d_sigmas, n_options * sizeof(float));
    cudaMalloc(&d_rights, n_options * sizeof(int));
    cudaMalloc(&d_vegas, n_options * sizeof(float));

    cudaMemcpy(d_spots, spots.data(), n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_strikes, strikes.data(), n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tenors, tenors.data(), n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sigmas, sigmas.data(), n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rights, rights_int.data(), n_options * sizeof(int), cudaMemcpyHostToDevice);

    if (n_divs > 0 && div_amounts.size() == div_times.size()) {
        cudaMalloc(&d_div_amounts, n_divs * sizeof(float));
        cudaMalloc(&d_div_times, n_divs * sizeof(float));
        cudaMemcpy(d_div_amounts, div_amounts.data(), n_divs * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_div_times, div_times.data(), n_divs * sizeof(float), cudaMemcpyHostToDevice);
    }

    int block_size = 256;
    int grid_size = (n_options + block_size - 1) / block_size;

    compute_vega_kernel_threadlocal<<<grid_size, block_size>>>(
        d_spots, d_strikes, d_tenors, d_sigmas, d_rights, r, n_steps, n_options,
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
    cudaFree(d_rights);
    cudaFree(d_vegas);
    if (d_div_amounts) cudaFree(d_div_amounts);
    if (d_div_times) cudaFree(d_div_times);

    return vegas;
}

__device__ inline float interp_linear_clamped(const float* X, const float* Y, int n, float xq) {
    if (xq <= X[0]) return Y[0];
    if (xq >= X[n - 1]) return Y[n - 1];
    int i = 0;
    for (; i < n - 1; ++i) {
        if (xq <= X[i + 1]) break;
    }
    float w = (xq - X[i]) / (X[i + 1] - X[i]);
    return Y[i] * (1.0f - w) + Y[i + 1] * w;
}

// Thomas tridiagonal solver: solves Ax = d; a=lower, b=diag, c=upper; n elements; in-place on d; x returned in out
__device__ void thomas_solve_inplace(const float* a, const float* b, const float* c,
                                     float* d, float* tmp_c, float* out, int n)
{
    // Forward sweep
    float c_star = c[0] / b[0];
    tmp_c[0] = c_star;
    d[0] = d[0] / b[0];
    for (int i = 1; i < n; ++i) {
        float m = 1.0f / (b[i] - a[i] * tmp_c[i - 1]);
        tmp_c[i] = (i == n - 1) ? 0.0f : c[i] * m;
        d[i] = (d[i] - a[i] * d[i - 1]) * m;
    }
    // Back substitution
    out[n - 1] = d[n - 1];
    for (int i = n - 2; i >= 0; --i) {
        out[i] = d[i] - tmp_c[i] * out[i + 1];
    }
}


// Enhanced accuracy FD pricer - better policy iteration and adaptive grid
// Not as correct as binomial tree so kinda useless.
__device__ float price_american_finite_diff_cash_div_threadlocal(
    float S, float K, float T,
    const float* r_curve, const float* time_points, int n_curve_points,
    float sigma, int is_call, int n_steps,
    const float* div_amounts, const int* div_steps, int n_divs)
{
    // Slightly larger grids for better accuracy
    const int MAX_SPACE = 121;   // Increased from 81
    const int MAX_TIME = 128;

    if (n_steps > MAX_TIME) n_steps = MAX_TIME;

    float dt = T / n_steps;
    if (dt <= 0.0f) {
        return is_call ? fmaxf(S - K, 0.0f) : fmaxf(K - S, 0.0f);
    }

    // Precompute rates
    float r_t[MAX_TIME];
    for (int step = 0; step < n_steps; ++step) {
        float t = (step + 1) * dt;
        if (n_curve_points <= 1) {
            r_t[step] = (n_curve_points == 1) ? r_curve[0] : 0.05f;
        } else {
            if (t <= time_points[0]) {
                r_t[step] = r_curve[0];
            } else if (t >= time_points[n_curve_points - 1]) {
                r_t[step] = r_curve[n_curve_points - 1];
            } else {
                int i = 0;
                for (; i < n_curve_points - 1; ++i) {
                    if (t <= time_points[i + 1]) break;
                }
                float w = (t - time_points[i]) / (time_points[i + 1] - time_points[i]);
                r_t[step] = r_curve[i] * (1.0f - w) + r_curve[i + 1] * w;
            }
        }
    }

    // Precompute dividends
    float D_at_step[MAX_TIME];
    for (int step = 0; step < n_steps; ++step) D_at_step[step] = 0.0f;
    for (int j = 0; j < n_divs; ++j) {
        int s = div_steps[j];
        if (s >= 1 && s <= n_steps) D_at_step[s - 1] += div_amounts[j];
    }

    // Better adaptive grid sizing - closer to slow version logic
    float vol_sqrtT = sigma * sqrtf(fmaxf(T, 1e-8f));
    float S_max = fmaxf(4.0f * K, S * expf(6.0f * vol_sqrtT));
    float S_min = 0.0f;

    // Use adaptive grid size like slow version
    int M = max(81, min(MAX_SPACE, n_steps * 2));
    if (M < 101) M = 101;  // Minimum size for accuracy
    float dS = (S_max - S_min) / M;

    // Arrays
    float S_grid[MAX_SPACE + 1];
    float V_old[MAX_SPACE + 1];
    float V_new[MAX_SPACE + 1];
    float payoff[MAX_SPACE + 1];

    // Store original coefficients for policy iteration
    float aL_orig[MAX_SPACE], aD_orig[MAX_SPACE], aU_orig[MAX_SPACE];
    float rhs_orig[MAX_SPACE];
    float aL[MAX_SPACE], aD[MAX_SPACE], aU[MAX_SPACE];
    float rhs[MAX_SPACE];

    // Initialize grid
    for (int i = 0; i <= M; ++i) {
        S_grid[i] = S_min + dS * i;
        payoff[i] = is_call ? fmaxf(S_grid[i] - K, 0.0f) : fmaxf(K - S_grid[i], 0.0f);
        V_old[i] = payoff[i];
    }

    // Backward in time
    for (int step = n_steps - 1; step >= 0; --step) {
        float r = r_t[step];
        float sig2 = sigma * sigma;

        // Build and store original Crank-Nicolson system
        for (int i = 1; i < M; ++i) {
            float S_i = S_grid[i];
            float S2 = S_i * S_i;

            float A = 0.25f * dt * (sig2 * S2 / (dS * dS) - r * S_i / dS);
            float B = -0.5f * dt * (sig2 * S2 / (dS * dS) + r);
            float C = 0.25f * dt * (sig2 * S2 / (dS * dS) + r * S_i / dS);

            aL_orig[i - 1] = -A;
            aD_orig[i - 1] = 1.0f - B;
            aU_orig[i - 1] = -C;
            rhs_orig[i - 1] = A * V_old[i - 1] + (1.0f + B) * V_old[i] + C * V_old[i + 1];
        }

        // Boundary conditions
        float V_left = is_call ? 0.0f : K;
        float V_right = is_call ? fmaxf(S_max - K, 0.0f) : 0.0f;
        rhs_orig[0] -= aL_orig[0] * V_left;
        rhs_orig[M - 2] -= aU_orig[M - 2] * V_right;

        // Policy iteration - more like the slow version
        const int max_policy_iter = 3;  // Increased from 1

        // Initial unconstrained solve
        for (int i = 0; i < M - 1; ++i) {
            aL[i] = aL_orig[i];
            aD[i] = aD_orig[i];
            aU[i] = aU_orig[i];
            rhs[i] = rhs_orig[i];
        }

        // Thomas solve
        float c_prime[MAX_SPACE];
        c_prime[0] = aU[0] / aD[0];
        rhs[0] = rhs[0] / aD[0];

        for (int i = 1; i < M - 1; ++i) {
            float m = aD[i] - aL[i] * c_prime[i - 1];
            c_prime[i] = (i == M - 2) ? 0.0f : aU[i] / m;
            rhs[i] = (rhs[i] - aL[i] * rhs[i - 1]) / m;
        }

        V_new[0] = V_left;
        V_new[M] = V_right;
        V_new[M - 1] = rhs[M - 2];
        for (int i = M - 3; i >= 0; --i) {
            V_new[i + 1] = rhs[i] - c_prime[i] * V_new[i + 2];
        }

        // Policy iterations like slow version
        for (int iter = 0; iter < max_policy_iter; ++iter) {
            // Check for violations
            bool any_violation = false;
            for (int i = 1; i < M; ++i) {
                if (V_new[i] < payoff[i]) {
                    any_violation = true;
                    break;
                }
            }

            if (!any_violation) break;

            // Rebuild system with violated points clamped
            for (int i = 1; i < M; ++i) {
                int k = i - 1;
                if (V_new[i] < payoff[i]) {
                    // Identity row for violated constraint
                    aL[k] = 0.0f;
                    aD[k] = 1.0f;
                    aU[k] = 0.0f;
                    rhs[k] = payoff[i];
                } else {
                    // Original CN coefficients
                    aL[k] = aL_orig[k];
                    aD[k] = aD_orig[k];
                    aU[k] = aU_orig[k];
                    rhs[k] = rhs_orig[k];
                }
            }

            // Boundary adjustments
            rhs[0] -= aL[0] * V_left;
            rhs[M - 2] -= aU[M - 2] * V_right;

            // Solve modified system
            c_prime[0] = (aD[0] != 0.0f) ? aU[0] / aD[0] : 0.0f;
            rhs[0] = (aD[0] != 0.0f) ? rhs[0] / aD[0] : 0.0f;

            for (int i = 1; i < M - 1; ++i) {
                float m = aD[i] - aL[i] * c_prime[i - 1];
                c_prime[i] = (i == M - 2) ? 0.0f : ((m != 0.0f) ? aU[i] / m : 0.0f);
                rhs[i] = (m != 0.0f) ? (rhs[i] - aL[i] * rhs[i - 1]) / m : 0.0f;
            }

            V_new[M - 1] = rhs[M - 2];
            for (int i = M - 3; i >= 0; --i) {
                V_new[i + 1] = rhs[i] - c_prime[i] * V_new[i + 2];
            }

            // Ensure American constraint is satisfied
            for (int i = 1; i < M; ++i) {
                if (V_new[i] < payoff[i]) {
                    V_new[i] = payoff[i];
                }
            }
        }

        // Enhanced dividend handling - more precise like slow version
        float D = D_at_step[step];
        if (D > 0.0f) {
            // Store current solution
            float V_temp[MAX_SPACE + 1];
            for (int i = 0; i <= M; ++i) V_temp[i] = V_new[i];

            // Apply dividend mapping with better interpolation
            for (int i = 0; i <= M; ++i) {
                float S_pre = S_grid[i];
                float S_post = fmaxf(S_pre - D, S_min);

                // More accurate interpolation
                if (S_post <= S_min) {
                    V_new[i] = V_temp[0];
                } else if (S_post >= S_max) {
                    V_new[i] = V_temp[M];
                } else {
                    // Find interpolation points
                    int j_low = (int)((S_post - S_min) / dS);
                    if (j_low < 0) j_low = 0;
                    if (j_low >= M) j_low = M - 1;

                    float weight = (S_post - S_grid[j_low]) / dS;
                    weight = fmaxf(0.0f, fminf(1.0f, weight));

                    V_new[i] = V_temp[j_low] * (1.0f - weight) + V_temp[j_low + 1] * weight;
                }

                // Enforce American constraint after dividend
                V_new[i] = fmaxf(V_new[i], payoff[i]);
            }
        }

        // Copy for next iteration
        for (int i = 0; i <= M; ++i) V_old[i] = V_new[i];
    }

    // Enhanced final interpolation
    if (S <= S_min) return V_old[0];
    if (S >= S_max) return V_old[M];

    float grid_pos = (S - S_min) / dS;
    int idx = (int)grid_pos;
    if (idx < 0) idx = 0;
    if (idx >= M) idx = M - 1;

    // Quadratic interpolation if we have enough points around S
    if (idx > 0 && idx < M - 1) {
        float x0 = S_grid[idx - 1], x1 = S_grid[idx], x2 = S_grid[idx + 1];
        float y0 = V_old[idx - 1], y1 = V_old[idx], y2 = V_old[idx + 1];

        // Lagrange quadratic interpolation
        float term1 = y0 * (S - x1) * (S - x2) / ((x0 - x1) * (x0 - x2));
        float term2 = y1 * (S - x0) * (S - x2) / ((x1 - x0) * (x1 - x2));
        float term3 = y2 * (S - x0) * (S - x1) / ((x2 - x0) * (x2 - x1));

        return term1 + term2 + term3;
    } else {
        // Linear interpolation at boundaries
        float weight = grid_pos - idx;
        return V_old[idx] * (1.0f - weight) + V_old[idx + 1] * weight;
    }
}

// Corrected and optimized: FD pricer (CN + Brennan–Schwartz policy iteration) with r(t) and cash dividends
// Implementation with smaller error compared to quantlib
__device__ float price_american_finite_diff_cash_div_threadlocal_slow(
    float S, float K, float T,
    const float* r_curve, const float* time_points, int n_curve_points,
    float sigma, int is_call, int n_steps,
    const float* div_amounts, const int* div_steps, int n_divs)
{
    constexpr int MAX_N = 512;      // time
    constexpr int MAX_M = 512;      // space
    if (n_steps > MAX_N) n_steps = MAX_N;

    float dt = T / n_steps;
    if (dt <= 0.0f) {
        float intrinsic = is_call ? fmaxf(S - K, 0.0f) : fmaxf(K - S, 0.0f);
        return intrinsic;
    }

    // Precompute r per time step (use upper time (step+1)*dt for better stability)
    float r_t[MAX_N];
    // linear interpolation helper
    auto r_at = [&](float t)->float {
        if (n_curve_points <= 0) return 0.0f;
        if (n_curve_points == 1) return r_curve[0];
        if (t <= time_points[0]) return r_curve[0];
        if (t >= time_points[n_curve_points - 1]) return r_curve[n_curve_points - 1];
        int i = 0;
        for (; i < n_curve_points - 1; ++i) { if (t <= time_points[i + 1]) break; }
        float w = (t - time_points[i]) / (time_points[i + 1] - time_points[i]);
        return r_curve[i] * (1.0f - w) + r_curve[i + 1] * w;
    };
    for (int step = 0; step < n_steps; ++step) {
        r_t[step] = r_at((step + 1) * dt);
    }

    // Precompute dividend per step (aggregate if multiple on same step)
    float D_at_step[MAX_N];
    for (int step = 0; step < n_steps; ++step) D_at_step[step] = 0.0f;
    for (int j = 0; j < n_divs; ++j) {
        int s = div_steps[j];
        if (s >= 1 && s <= n_steps) D_at_step[s - 1] += div_amounts[j];
    }

    // Space grid sizing: smaller and adaptive
    float vol_sqrtT = sigma * sqrtf(fmaxf(T, 1e-8f));
    float S_max = fmaxf(4.0f * K, S * expf(6.0f * vol_sqrtT));
    float S_min = 0.0f;

    int M = max(81, min(MAX_M, n_steps * 2)); // keep modest to avoid local memory blow-up
    float dS = (S_max - S_min) / M;

    // Grids
    float S_grid[MAX_M + 1];
    float V_old[MAX_M + 1];
    float V_new[MAX_M + 1];
    float payoff[MAX_M + 1];

    // Tridiagonal buffers for interior points (size M-1)
    float aL[MAX_M - 1], aD[MAX_M - 1], aU[MAX_M - 1];
    float rhs[MAX_M - 1], tmp_c[MAX_M - 1], sol[MAX_M - 1];

    for (int i = 0; i <= M; ++i) {
        S_grid[i] = S_min + dS * i;
        payoff[i] = is_call ? fmaxf(S_grid[i] - K, 0.0f) : fmaxf(K - S_grid[i], 0.0f);
        V_old[i]  = payoff[i];
    }

    // Backward in time
    const int max_policy_iter = 5;    // usually 1-3 suffice
    for (int step = n_steps - 1; step >= 0; --step) {
        float r = r_t[step];
        float sig2 = sigma * sigma;

        // Build RHS using CN
        for (int i = 1; i < M; ++i) {
            float S_i = S_grid[i];
            float S2  = S_i * S_i;

            float A = 0.25f * dt * (sig2 * S2 / (dS * dS) - r * S_i / dS);
            float B = -0.5f * dt * (sig2 * S2 / (dS * dS) + r);
            float C = 0.25f * dt * (sig2 * S2 / (dS * dS) + r * S_i / dS);

            // LHS (A^*) diagonal and off-diagonals
            aL[i - 1] = -A;
            aD[i - 1] = 1.0f - B;
            aU[i - 1] = -C;

            rhs[i - 1] = A * V_old[i - 1] + (1.0f + B) * V_old[i] + C * V_old[i + 1];
        }

        // Boundary conditions
        float V_left  = is_call ? 0.0f : K;
        float V_right = is_call ? fmaxf(S_max - K, 0.0f) : 0.0f;
        // Adjust RHS for boundaries
        rhs[0]       -= aL[0]           * V_left;
        rhs[M - 2]   -= aU[M - 2]       * V_right;

        // Policy iteration: enforce American constraint by modifying rows violating payoff
        // Start with the unconstrained solve
        // We’ll do up to max_policy_iter passes; each pass solves with rows fixed at payoff where violated.
        // We implement row-fixing by overwriting aD and rhs row-wise temporarily.
        // Keep original aD/rhs in scratch if needed; instead, we detect violations and do a second Thomas with rows clamped.
        // First: unconstrained solve
        for (int i = 0; i < M - 1; ++i) { sol[i] = rhs[i]; } // reuse sol as RHS input
        thomas_solve_inplace(aL, aD, aU, sol, tmp_c, sol, M - 1);

        // Map back to V_new with boundaries
        V_new[0] = V_left;
        for (int i = 1; i < M; ++i) V_new[i] = sol[i - 1];
        V_new[M] = V_right;

        // Apply American constraint and iterate a few times if needed
        for (int iter = 0; iter < max_policy_iter; ++iter) {
            // Detect violations
            int any_violation = 0;
            for (int i = 1; i < M; ++i) {
                if (V_new[i] < payoff[i]) { any_violation = 1; break; }
            }
            if (!any_violation) break;

            // Build modified system: clamp violating rows to identity
            // We create modified RHS/diag in-place; reuse arrays
            for (int i = 1; i < M; ++i) {
                int k = i - 1;
                if (V_new[i] < payoff[i]) {
                    // Identity row: x_i = payoff_i
                    // Implement by setting diag large and RHS large*payoff_i
                    // Instead of large numbers, do exact identity by splitting the solve:
                    // We’ll mark this row and solve via two sweeps (cheap approximation):
                    aL[k] = 0.0f;
                    aD[k] = 1.0f;
                    aU[k] = 0.0f;
                    rhs[k] = payoff[i];
                } else {
                    float S_i = S_grid[i];
                    float S2  = S_i * S_i;
                    float A = 0.25f * dt * (sig2 * S2 / (dS * dS) - r * S_i / dS);
                    float B = -0.5f * dt * (sig2 * S2 / (dS * dS) + r);
                    float C = 0.25f * dt * (sig2 * S2 / (dS * dS) + r * S_i / dS);
                    aL[k] = -A;
                    aD[k] = 1.0f - B;
                    aU[k] = -C;
                    rhs[k] = A * V_old[i - 1] + (1.0f + B) * V_old[i] + C * V_old[i + 1];
                }
            }
            // Boundary adjustments
            rhs[0]     -= aL[0]       * V_left;
            rhs[M - 2] -= aU[M - 2]   * V_right;

            // Solve again
            for (int i = 0; i < M - 1; ++i) { sol[i] = rhs[i]; }
            thomas_solve_inplace(aL, aD, aU, sol, tmp_c, sol, M - 1);
            V_new[0] = V_left;
            for (int i = 1; i < M; ++i) V_new[i] = sol[i - 1];
            V_new[M] = V_right;
        }

        // Dividend mapping if any at this step: from post-div to pre-div V
        float D = D_at_step[step];
        if (D > 0.0f) {
            // Copy V_new to V_old temporarily
            for (int i = 0; i <= M; ++i) V_old[i] = V_new[i];
            for (int i = 0; i <= M; ++i) {
                float S_pre  = S_grid[i];
                float S_post = fmaxf(S_pre - D, S_min);
                V_new[i] = interp_linear_clamped(S_grid, V_old, M + 1, S_post);
                if (V_new[i] < payoff[i]) V_new[i] = payoff[i];
            }
        }

        // Prepare for next step
        for (int i = 0; i <= M; ++i) V_old[i] = V_new[i];
    }

    // Interpolate final value at S
    return interp_linear_clamped(S_grid, V_old, M + 1, S);
}

// Faster IV: fewer iterations with tighter bracket and relative tolerance
__device__ float implied_vol_american_fd_bisection_threadlocal(
    float target, float S, float K, float T,
    const float* r_curve, const float* time_points, int n_curve_points,
    int is_call, int n_steps,
    const float* div_amounts, const int* div_steps, int n_divs,
    float tol = 5e-5f, int max_iter = 30, float v_min = 1e-4f, float v_max = 4.0f)
{
    // Expand bracket adaptively until it contains target
    float lo = v_min, hi = v_max;
    float p_lo = price_american_finite_diff_cash_div_threadlocal(S, K, T, r_curve, time_points, n_curve_points,
                                                                 lo, is_call, n_steps, div_amounts, div_steps, n_divs);
    float p_hi = price_american_finite_diff_cash_div_threadlocal(S, K, T, r_curve, time_points, n_curve_points,
                                                                 hi, is_call, n_steps, div_amounts, div_steps, n_divs);
    int expand = 0;
    while (target > p_hi && expand < 3) {
        hi *= 1.5f;
        p_hi = price_american_finite_diff_cash_div_threadlocal(S, K, T, r_curve, time_points, n_curve_points,
                                                               hi, is_call, n_steps, div_amounts, div_steps, n_divs);
        ++expand;
    }
    if (target <= p_lo + 1e-8f) return 0.0f;
    if (target > p_hi + 1e-8f) return NAN;

    float mid = 0.0f;
    for (int i = 0; i < max_iter; ++i) {
        mid = 0.5f * (lo + hi);
        float p_mid = price_american_finite_diff_cash_div_threadlocal(S, K, T, r_curve, time_points, n_curve_points,
                                                                      mid, is_call, n_steps, div_amounts, div_steps, n_divs);
        float diff = p_mid - target;
        if (fabsf(diff) <= fmaxf(tol, 1e-4f * target)) return mid;
        if (diff < 0.0f) lo = mid; else hi = mid;
        if (hi - lo < 1e-5f) break;
    }
    return mid;
}


// New IV kernel using corrected FD pricer with time-dependent rates (signature unchanged)
__global__ void compute_iv_fd_kernel_threadlocal(
    const float* prices, const float* spots, const float* strikes, const float* tenors,
    const int* rights, const float* r_curve, const float* time_points, int n_curve_points,
    int n_steps, int n_options,
    const float* div_amounts, const float* div_times, int n_divs,
    float* results)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_options) return;

    float p = prices[idx];
    float s = spots[idx];
    float k = strikes[idx];
    float t = tenors[idx];
    int right = rights[idx];

    if (!isfinite(p) || !isfinite(s) || !isfinite(k) || !isfinite(t) ||
        t <= 0.0f || s <= 0.0f || k <= 0.0f || p <= 0.0f) {
        results[idx] = (t <= 0.0f) ? 0.0f : NAN;
        return;
        }

    // Only include dividends before expiry
    const int MAX_DIVS = 32;
    float local_div_amounts[MAX_DIVS];
    int   local_div_steps[MAX_DIVS];
    int valid_divs = 0;

    float intrinsic = (right == 1) ? fmaxf(s - k, 0.0f) : fmaxf(k - s, 0.0f);
    if (fabsf(p - intrinsic) < 1e-6f) {
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

    results[idx] = implied_vol_american_fd_bisection_threadlocal(
        p, s, k, t, r_curve, time_points, n_curve_points, right, n_steps,
        local_div_amounts, local_div_steps, valid_divs
    );
}


// Host wrapper for new finite difference IV calculation with time-dependent rates
std::vector<float> get_v_iv_fd_with_term_structure_cuda(
    const std::vector<float>& prices,
    const std::vector<float>& spots,
    const std::vector<float>& strikes,
    const std::vector<float>& tenors,
    const std::vector<std::string>& rights,
    const std::vector<float>& r_curve,
    const std::vector<float>& time_points,
    int n_steps = 200,
    const std::vector<float>& div_amounts = {},
    const std::vector<float>& div_times = {}) {

    int n_options = prices.size();
    std::vector<float> results(n_options);

    if (n_options == 0) return results;

    if (r_curve.size() != time_points.size()) {
        throw std::invalid_argument("r_curve and time_points must have the same size");
    }

    // Convert rights to int
    std::vector<int> rights_int(n_options);
    for (int i = 0; i < n_options; ++i) {
        rights_int[i] = (rights[i] == "c" || rights[i] == "C" ||
                        rights[i] == "call" || rights[i] == "CALL") ? 1 : 0;
    }

    int n_divs = div_amounts.size();
    int n_curve_points = r_curve.size();

    // Allocate device memory
    float *d_prices, *d_spots, *d_strikes, *d_tenors, *d_results;
    int *d_rights;
    float *d_div_amounts = nullptr, *d_div_times = nullptr;
    float *d_r_curve, *d_time_points;

    cudaMalloc(&d_prices, n_options * sizeof(float));
    cudaMalloc(&d_spots, n_options * sizeof(float));
    cudaMalloc(&d_strikes, n_options * sizeof(float));
    cudaMalloc(&d_tenors, n_options * sizeof(float));
    cudaMalloc(&d_rights, n_options * sizeof(int));
    cudaMalloc(&d_results, n_options * sizeof(float));
    cudaMalloc(&d_r_curve, n_curve_points * sizeof(float));
    cudaMalloc(&d_time_points, n_curve_points * sizeof(float));

    cudaMemcpy(d_prices, prices.data(), n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_spots, spots.data(), n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_strikes, strikes.data(), n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tenors, tenors.data(), n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rights, rights_int.data(), n_options * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_r_curve, r_curve.data(), n_curve_points * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_time_points, time_points.data(), n_curve_points * sizeof(float), cudaMemcpyHostToDevice);

    if (n_divs > 0 && div_amounts.size() == div_times.size()) {
        cudaMalloc(&d_div_amounts, n_divs * sizeof(float));
        cudaMalloc(&d_div_times, n_divs * sizeof(float));
        cudaMemcpy(d_div_amounts, div_amounts.data(), n_divs * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_div_times, div_times.data(), n_divs * sizeof(float), cudaMemcpyHostToDevice);
    }

    // Launch kernel
    int block_size = 256;
    int grid_size = (n_options + block_size - 1) / block_size;

    compute_iv_fd_kernel_threadlocal<<<grid_size, block_size>>>(
        d_prices, d_spots, d_strikes, d_tenors, d_rights, d_r_curve, d_time_points, n_curve_points,
        n_steps, n_options, d_div_amounts, d_div_times, n_divs, d_results
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));

    cudaMemcpy(results.data(), d_results, n_options * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_prices);
    cudaFree(d_spots);
    cudaFree(d_strikes);
    cudaFree(d_tenors);
    cudaFree(d_rights);
    cudaFree(d_results);
    cudaFree(d_r_curve);
    cudaFree(d_time_points);
    if (d_div_amounts) cudaFree(d_div_amounts);
    if (d_div_times) cudaFree(d_div_times);

    return results;
}
