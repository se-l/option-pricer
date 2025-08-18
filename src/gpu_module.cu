
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <vector>
#include <stdexcept>

namespace py = pybind11;

// =======================
// Major Fix: All tree/temp arrays are per-thread, local stack arrays, not shared_mem
// =======================

__device__ float price_american_binomial_cash_div_threadlocal(
    float S, float K, float T, float r, float sigma, int is_call, int n_steps,
    const float* div_amounts, const int* div_steps, int n_divs)
{
    constexpr int MAX_STEPS = 512;
    float V[MAX_STEPS + 1];
    float stock[MAX_STEPS + 1];

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

    // 1. Build the vector of stock prices at maturity, with all dividends applied.
    // We'll do this per-path for each end node.
    for (int j = 0; j <= N; ++j) {
        float S_j = S;
        int step_idx = 0;
        int n_up = j;
        int n_down = N - j;
        // Build the actual exercise path: Up N times, Down N-j times in some order.
        // The step at which dividend occurs depends on step
        // Because dividend is at discrete step, we must apply at steps = div_steps[i]
        // For each step, know how many ups happened so far:
        // Since order doesn't matter for stock, we can do the math up front.
        S_j *= powf(u, n_up) * powf(d, n_down);

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
                float S_before = S * powf(u, ups_before_div) * powf(d, downs_before_div);
                // At this node, subtract the dividend (apply at step_div)
                S_j -= div_amounts[div_i] * powf(u, j - ups_before_div) * powf(d, (N - j) - downs_before_div);
                // Stock can not become negative
                if (S_j < 1e-8f) S_j = 1e-8f;
            }
        }
        stock[j] = S_j;
        V[j] = is_call ? fmaxf(S_j - K, 0.0f) : fmaxf(K - S_j, 0.0f);
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
                        float S_before = S * powf(u, ups_before_div) * powf(d, downs_before_div);
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

// =======================
// Main IV kernel: no shared memory needed
// =======================

__global__ void compute_iv_kernel_threadlocal(
    const float* prices, const float* spots, const float* strikes, const float* tenors,
    const int* rights, float r, int n_steps, int n_options,
    const float* div_amounts, const float* div_times, int n_divs,
    float* results) {

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
        local_div_amounts, local_div_steps, valid_divs
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
    int n_steps = 100,
    const std::vector<float>& div_amounts = {},
    const std::vector<float>& div_times = {}) {

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
        d_div_amounts, d_div_times, n_divs, d_results
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

// Update Python binding
PYBIND11_MODULE(gpu_module, m) {
    m.doc() = "CUDA-accelerated American option implied volatility calculator";

    // New simplified interface with single dividend schedule
    m.def("get_v_iv_fd_single_underlying", &get_v_iv_fd_cuda,
          "Compute implied volatilities with single dividend schedule for all options",
          py::arg("prices"), py::arg("spots"), py::arg("strikes"), py::arg("tenors"), py::arg("rights"),
          py::arg("r"), py::arg("n_steps") = 100,
          py::arg("div_amounts") = std::vector<float>(),
          py::arg("div_times") = std::vector<float>());
}