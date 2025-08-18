
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <vector>
#include <stdexcept>

namespace py = pybind11;

// CUDA kernel for American binomial option pricing with cash dividends
__device__ float price_american_binomial_cash_div(
    float S, float K, float T, float r, float sigma, int is_call, int n_steps,
    const float* div_amounts, const int* div_steps, int n_divs) {

    // Adjust steps based on tenor to maintain numerical stability
    int effective_steps = n_steps;
    if (T < 0.1f) {
        // For tenors < 0.1 years (36 days), use fewer steps
        effective_steps = (int)(T * n_steps * 10); // Scale with tenor
        effective_steps = max(10, min(effective_steps, n_steps));
    }

    float dt = T / effective_steps;

    // Add stricter bounds checking
    if (dt <= 0.0f || dt > 0.01f) {  // Max 3.65 days per step
        return is_call ? fmaxf(S - K, 0.0f) : fmaxf(K - S, 0.0f);
    }

    float u = expf(sigma * sqrtf(dt));
    if (u <= 1.001f) {  // More conservative bound
        u = 1.001f;
    }

    if (u <= 1.0f) u = 1.000001f;

    // Add minimum time step validation
    if (dt < 1e-6f) {
        // For very short tenors, fall back to intrinsic value or use fewer steps
        return is_call ? fmaxf(S - K, 0.0f) : fmaxf(K - S, 0.0f);
    }


    float d = 1.0f / u;
    float edtr = expf(r * dt);
    float p = (edtr - d) / (u - d);

    // Add strict probability bounds - this is crucial
    if (p <= 0.0f || p >= 1.0f || !isfinite(p)) {
        // Numerical instability detected
        return is_call ? fmaxf(S - K, 0.0f) : fmaxf(K - S, 0.0f);
    }

    p = fmaxf(0.0f, fminf(1.0f, p));
    float disc = expf(-r * dt);

    extern __shared__ float shared_mem[];

    // Each thread gets its own section of shared memory
    int thread_id = threadIdx.x;
    int arrays_per_thread = 3 * (n_steps + 1);
    int thread_offset = thread_id * arrays_per_thread;

    float* V_level = shared_mem + thread_offset;
    float* S_level = shared_mem + thread_offset + (n_steps + 1);
    float* next_S_level = shared_mem + thread_offset + 2 * (n_steps + 1);


    // Initialize at t=0
    S_level[0] = S;
    int level_size = 1;
    int div_idx = 0;

    // Forward evolution to maturity
    for (int step = 1; step <= n_steps; ++step) {
        // Build next level
        next_S_level[0] = S_level[0] * d;
        for (int j = 1; j < level_size; ++j) {
            next_S_level[j] = S_level[j - 1] * u;
        }
        next_S_level[level_size] = S_level[level_size - 1] * u;
        level_size++;

        // Apply dividends at this step
        if (n_divs > 0) {
            float div_sum = 0.0f;
            while (div_idx < n_divs && div_steps[div_idx] == step) {
                div_sum += div_amounts[div_idx];
                div_idx++;
            }
            if (div_sum != 0.0f) {
                for (int j = 0; j < level_size; ++j) {
                    float val = next_S_level[j] - div_sum;
                    next_S_level[j] = fmaxf(val, 1.0e-10f);
                }
            }
        }

        // Copy to current level
        for (int j = 0; j < level_size; ++j) {
            S_level[j] = next_S_level[j];
        }
    }

    // Terminal payoffs
    for (int j = 0; j < level_size; ++j) {
        V_level[j] = is_call ? fmaxf(S_level[j] - K, 0.0f) : fmaxf(K - S_level[j], 0.0f);
    }

    // Backward induction with early exercise
    for (int step = n_steps - 1; step >= 0; --step) {
        // Roll back option values
        for (int j = 0; j <= step; ++j) {
            float cont = disc * (p * V_level[j + 1] + (1.0f - p) * V_level[j]);
            V_level[j] = cont;
        }

        // Rebuild S_level for this step (simplified reconstruction)
        level_size = step + 1;
        S_level[0] = S;
        int current_level = 1;
        int div_idx2 = 0;

        for (int st = 1; st <= step + 1; ++st) {
            // Build next level from current
            next_S_level[0] = S_level[0] * d;
            for (int jj = 1; jj < current_level; ++jj) {
                next_S_level[jj] = S_level[jj - 1] * u;
            }
            next_S_level[current_level] = S_level[current_level - 1] * u;
            current_level++;

            // Apply dividends
            if (n_divs > 0) {
                float div_sum2 = 0.0f;
                for (int kdiv = 0; kdiv < n_divs; ++kdiv) {
                    if (div_steps[kdiv] == st) {
                        div_sum2 += div_amounts[kdiv];
                    }
                }
                if (div_sum2 != 0.0f) {
                    for (int jj = 0; jj < current_level; ++jj) {
                        float val = next_S_level[jj] - div_sum2;
                        next_S_level[jj] = fmaxf(val, 1.0e-10f);
                    }
                }
            }

            // Copy back
            for (int jj = 0; jj < current_level; ++jj) {
                S_level[jj] = next_S_level[jj];
            }
        }

        // Early exercise check
        for (int j = 0; j < level_size; ++j) {
            float ex = is_call ? fmaxf(S_level[j] - K, 0.0f) : fmaxf(K - S_level[j], 0.0f);
            V_level[j] = fmaxf(V_level[j], ex);
        }
    }

    return V_level[0];
}

// CUDA kernel for implied volatility bisection
__device__ float implied_vol_american_bisection(
    float target, float S, float K, float T, float r, int is_call, int n_steps,
    const float* div_amounts, const int* div_steps, int n_divs,
    float tol = 1e-6f, int max_iter = 100, float v_min = 1e-4f, float v_max = 5.0f) {

    float p_low = price_american_binomial_cash_div(S, K, T, r, v_min, is_call, n_steps, div_amounts, div_steps, n_divs);
    float p_high = price_american_binomial_cash_div(S, K, T, r, v_max, is_call, n_steps, div_amounts, div_steps, n_divs);

    if (target <= p_low + 1e-12f) return 0.0f;
    if (target > p_high + 1e-12f) return NAN;

    float lo = v_min, hi = v_max, mid = 0.0f;
    for (int i = 0; i < max_iter; ++i) {
        mid = 0.5f * (lo + hi);
        float p_mid = price_american_binomial_cash_div(S, K, T, r, mid, is_call, n_steps, div_amounts, div_steps, n_divs);
        float diff = p_mid - target;

        if (fabsf(diff) < tol) return mid;
        if (diff < 0.0f) lo = mid;
        else hi = mid;
        if (hi - lo < tol) break;
    }
    return mid;
}

// Main CUDA kernel for vectorized IV calculation
__global__ void compute_iv_kernel(
    const float* prices, const float* spots, const float* strikes, const float* tenors,
    const int* rights, float r, int n_steps, int n_options,
    const float* div_amounts_concat, const float* div_times_concat,
    const int* div_index_start, const int* div_count,
    float* results) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_options) return;

    float p = prices[idx];
    float s = spots[idx];
    float k = strikes[idx];
    float t = tenors[idx];
    int right = rights[idx];

    // Validate inputs
    if (!isfinite(p) || !isfinite(s) || !isfinite(k) || !isfinite(t) ||
        t <= 0.0f || s <= 0.0f || k <= 0.0f || p <= 0.0f) {
        results[idx] = (t <= 0.0f) ? 0.0f : NAN;
        return;
    }

    // Extract dividends for this option
    int start = div_index_start[idx];
    int cnt = div_count[idx];

    // Allocate local arrays for dividends (max reasonable size)
    const int MAX_DIVS = 20;
    float local_div_amounts[MAX_DIVS];
    int local_div_steps[MAX_DIVS];
    int valid_divs = 0;

    if (cnt > 0 && cnt <= MAX_DIVS) {
        for (int j = 0; j < cnt; ++j) {
            float div_time = div_times_concat[start + j];
            if (div_time > 0.0f && div_time < t) {
                int step = (int)roundf(div_time / t * n_steps);
                step = max(1, min(step, n_steps));
                local_div_steps[valid_divs] = step;
                local_div_amounts[valid_divs] = div_amounts_concat[start + j];
                valid_divs++;
            }
        }
    }

    // Call implied volatility solver
    results[idx] = implied_vol_american_bisection(
        p, s, k, t, r, right, n_steps,
        local_div_amounts, local_div_steps, valid_divs
    );
}

// Host function to launch CUDA kernel
std::vector<float> get_v_iv_fd_cuda(
    const std::vector<float>& prices,
    const std::vector<float>& spots,
    const std::vector<float>& strikes,
    const std::vector<float>& tenors,
    const std::vector<std::string>& rights,
    float r,
    int n_steps = 100,
    const std::vector<float>& div_amounts_concat = {},
    const std::vector<float>& div_times_concat = {},
    const std::vector<int>& div_index_start = {},
    const std::vector<int>& div_count = {}) {

    int n_options = prices.size();
    std::vector<float> results(n_options);

    if (n_options == 0) return results;

    // Convert rights to integers
    std::vector<int> rights_int(n_options);
    for (int i = 0; i < n_options; ++i) {
        rights_int[i] = (rights[i] == "c" || rights[i] == "C" ||
                        rights[i] == "call" || rights[i] == "CALL") ? 1 : 0;
    }

    // Allocate device memory
    float *d_prices, *d_spots, *d_strikes, *d_tenors, *d_results;
    int *d_rights;
    float *d_div_amounts = nullptr, *d_div_times = nullptr;
    int *d_div_start = nullptr, *d_div_count = nullptr;

    cudaMalloc(&d_prices, n_options * sizeof(float));
    cudaMalloc(&d_spots, n_options * sizeof(float));
    cudaMalloc(&d_strikes, n_options * sizeof(float));
    cudaMalloc(&d_tenors, n_options * sizeof(float));
    cudaMalloc(&d_rights, n_options * sizeof(int));
    cudaMalloc(&d_results, n_options * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_prices, prices.data(), n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_spots, spots.data(), n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_strikes, strikes.data(), n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tenors, tenors.data(), n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rights, rights_int.data(), n_options * sizeof(int), cudaMemcpyHostToDevice);

    // Handle dividends if provided
    if (!div_amounts_concat.empty() && !div_times_concat.empty() &&
        !div_index_start.empty() && !div_count.empty()) {
        cudaMalloc(&d_div_amounts, div_amounts_concat.size() * sizeof(float));
        cudaMalloc(&d_div_times, div_times_concat.size() * sizeof(float));
        cudaMalloc(&d_div_start, div_index_start.size() * sizeof(int));
        cudaMalloc(&d_div_count, div_count.size() * sizeof(int));

        cudaMemcpy(d_div_amounts, div_amounts_concat.data(),
                  div_amounts_concat.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_div_times, div_times_concat.data(),
                  div_times_concat.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_div_start, div_index_start.data(),
                  div_index_start.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_div_count, div_count.data(),
                  div_count.size() * sizeof(int), cudaMemcpyHostToDevice);
    }

    // Launch kernel
    int block_size = 256;
    int grid_size = (n_options + block_size - 1) / block_size;
    int shared_mem_size = 3 * (n_steps + 1) * sizeof(float); // V_level, S_level, next_S_level

    compute_iv_kernel<<<grid_size, block_size, shared_mem_size>>>(
        d_prices, d_spots, d_strikes, d_tenors, d_rights, r, n_steps, n_options,
        d_div_amounts, d_div_times, d_div_start, d_div_count, d_results
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }

    // Copy results back
    cudaMemcpy(results.data(), d_results, n_options * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_prices);
    cudaFree(d_spots);
    cudaFree(d_strikes);
    cudaFree(d_tenors);
    cudaFree(d_rights);
    cudaFree(d_results);
    if (d_div_amounts) cudaFree(d_div_amounts);
    if (d_div_times) cudaFree(d_div_times);
    if (d_div_start) cudaFree(d_div_start);
    if (d_div_count) cudaFree(d_div_count);

    return results;
}


// Updated CUDA kernel for vectorized IV calculation with shared dividend schedule
__global__ void compute_iv_kernel_v2(
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

    // Validate inputs
    if (!isfinite(p) || !isfinite(s) || !isfinite(k) || !isfinite(t) ||
        t <= 0.0f || s <= 0.0f || k <= 0.0f || p <= 0.0f) {
        results[idx] = (t <= 0.0f) ? 0.0f : NAN;
        return;
    }

    // Filter dividends relevant to this option's expiry
    const int MAX_DIVS = 20;
    float local_div_amounts[MAX_DIVS];
    int local_div_steps[MAX_DIVS];
    int valid_divs = 0;

    // Only include dividends that occur before this option's expiry
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

    // Call implied volatility solver
    results[idx] = implied_vol_american_bisection(
        p, s, k, t, r, right, n_steps,
        local_div_amounts, local_div_steps, valid_divs
    );
}

// Simplified host function with single dividend schedule for all options
std::vector<float> get_v_iv_fd_cuda_v2(
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

    // Convert rights to integers
    std::vector<int> rights_int(n_options);
    for (int i = 0; i < n_options; ++i) {
        rights_int[i] = (rights[i] == "c" || rights[i] == "C" ||
                        rights[i] == "call" || rights[i] == "CALL") ? 1 : 0;
    }

    int n_divs = div_amounts.size();

    // Allocate device memory
    float *d_prices, *d_spots, *d_strikes, *d_tenors, *d_results;
    int *d_rights;
    float *d_div_amounts = nullptr, *d_div_times = nullptr;

    cudaMalloc(&d_prices, n_options * sizeof(float));
    cudaMalloc(&d_spots, n_options * sizeof(float));
    cudaMalloc(&d_strikes, n_options * sizeof(float));
    cudaMalloc(&d_tenors, n_options * sizeof(float));
    cudaMalloc(&d_rights, n_options * sizeof(int));
    cudaMalloc(&d_results, n_options * sizeof(float));

    // Copy main data to device
    cudaMemcpy(d_prices, prices.data(), n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_spots, spots.data(), n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_strikes, strikes.data(), n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tenors, tenors.data(), n_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rights, rights_int.data(), n_options * sizeof(int), cudaMemcpyHostToDevice);

    // Handle dividend schedule (shared across all options)
    if (n_divs > 0 && div_amounts.size() == div_times.size()) {
        cudaMalloc(&d_div_amounts, n_divs * sizeof(float));
        cudaMalloc(&d_div_times, n_divs * sizeof(float));

        cudaMemcpy(d_div_amounts, div_amounts.data(), n_divs * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_div_times, div_times.data(), n_divs * sizeof(float), cudaMemcpyHostToDevice);
    }

    // Launch kernel
    int block_size = 256;
    int grid_size = (n_options + block_size - 1) / block_size;
    int shared_mem_size = 3 * (n_steps + 1) * sizeof(float);

    compute_iv_kernel_v2<<<grid_size, block_size, shared_mem_size>>>(
        d_prices, d_spots, d_strikes, d_tenors, d_rights, r, n_steps, n_options,
        d_div_amounts, d_div_times, n_divs, d_results
    );

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }

    // Copy results back
    cudaMemcpy(results.data(), d_results, n_options * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
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
    m.def("get_v_iv_fd_single_underlying", &get_v_iv_fd_cuda_v2,
          "Compute implied volatilities with single dividend schedule for all options",
          py::arg("prices"), py::arg("spots"), py::arg("strikes"), py::arg("tenors"), py::arg("rights"),
          py::arg("r"), py::arg("n_steps") = 100,
          py::arg("div_amounts") = std::vector<float>(),
          py::arg("div_times") = std::vector<float>());

    m.def("get_v_iv_fd", &get_v_iv_fd_cuda, "Compute implied volatilities for American options using CUDA",
          py::arg("prices"), py::arg("spots"), py::arg("strikes"), py::arg("tenors"), py::arg("rights"),
          py::arg("r"), py::arg("n_steps") = 100,
          py::arg("div_amounts_concat") = std::vector<float>(),
          py::arg("div_times_concat") = std::vector<float>(),
          py::arg("div_index_start") = std::vector<int>(),
          py::arg("div_count") = std::vector<int>());
}