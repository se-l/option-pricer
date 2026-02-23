#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <vector>
#include "../../src/core/pricing_engine.h"

TEST_CASE("Consistency: CPU vs GPU FD Pricing", "[pricing][cuda]") {
    // 1. Setup Test Grid
    std::vector<uint8_t> rights = {1, 0}; // Call, Put
    std::vector tenors = {0.1f, 0.5f, 2.0f}; // 1m, 6m, 2y
    std::vector strikes = {80.0f, 90.0f, 100.0f, 110.0f, 120.0f};

    float S = 100.0f;
    float sigma = 0.25f;

    std::vector rates_curve = {0.03f, 0.1f, 0.03f};
    std::vector rates_times = {0.2f, 0.8f, 1.5f};
    std::vector div_amounts = {1.0f, 1.50f, 2.0f};
    std::vector div_times = {0.4f, 0.8f, 1.2f};
    // std::vector rates_curve = {0.0f, 0.0f};
    // std::vector rates_times = {0.0f, 2.0f};
    // std::vector div_amounts = {0.0f};
    // std::vector div_times = {0.0f};

    // 2. Prepare Vectorized Inputs
    std::vector<float> v_spots, v_strikes, v_tenors, v_sigmas;
    std::vector<uint8_t> v_rights;

    for (auto r : rights) {
        for (auto t : tenors) {
            for (auto k : strikes) {
                v_spots.push_back(S);
                v_strikes.push_back(k);
                v_tenors.push_back(t);
                v_sigmas.push_back(sigma);
                v_rights.push_back(r);
            }
        }
    }

    // // 3. Run GPU Vectorized Version
    std::vector<float> gpu_results = get_v_price_fd_cuda(
        v_spots, v_strikes, v_tenors, v_sigmas, v_rights,
        rates_curve, rates_times,
        div_amounts, div_times,
        200, 200
    );

    // Verify GPU returned the expected number of results
    REQUIRE(gpu_results.size() == v_spots.size());

    // 4. Compare with CPU Results
    const float tolerance = 1e-3f;

    for (size_t i = 0; i < v_spots.size(); ++i) {
        float cpu_price = price_american_fd_div_host(
            v_spots[i], v_strikes[i], v_tenors[i], v_sigmas[i], v_rights[i],
            rates_curve, rates_times,
            div_amounts, div_times,
            200, 200
        );

        float gpu_price = gpu_results[i];

        // DYNAMIC_SECTION allows Catch2 to report which specific option failed
        std::string type = v_rights[i] ? "Call" : "Put";
        DYNAMIC_SECTION("Option " << type << " T=" << v_tenors[i] << " K=" << v_strikes[i]) {
            CHECK_THAT(gpu_price, Catch::Matchers::WithinRel(cpu_price, tolerance) || Catch::Matchers::WithinAbs(0, tolerance));
        }
    }
}

TEST_CASE("Robustness: get_v_fd_price_cuda with extreme arguments", "[pricing][cuda]") {
    // Define extreme input sets: vary S, K, T, sigma, is_call
    // Include edge cases like very small/large values, zeros, negatives (where applicable, though function guards against invalid)
    struct TestCase {
        float S, K, T, sigma;
        uint8_t is_call;
        std::vector<float> rates_curve, rates_times, div_amounts, div_times;
        int time_steps, space_steps;
    };

    std::vector<TestCase> cases = {
        // Near-zero values
        {1e-6f, 100.0f, 1.0f, 0.2f, 1, {0.05f}, {1.0f}, {1.0f}, {0.5f}, 200, 200},  // Tiny S
        {100.0f, 1e-6f, 1.0f, 0.2f, 0, {0.05f}, {1.0f}, {}, {}, 200, 200},          // Tiny K (put)
        {100.0f, 100.0f, 1e-6f, 0.2f, 1, {0.05f}, {1.0f}, {}, {}, 200, 200},        // Tiny T
        {100.0f, 100.0f, 1.0f, 1e-6f, 1, {0.05f}, {1.0f}, {}, {}, 200, 200},        // Tiny sigma

        // Very large values
        {1e6f, 100.0f, 1.0f, 0.2f, 1, {0.05f}, {1.0f}, {1000.0f}, {0.5f}, 200, 200},  // Large S (call)
        {100.0f, 1e6f, 1.0f, 0.2f, 0, {0.05f}, {1.0f}, {}, {}, 200, 200},             // Large K (put)
        {100.0f, 100.0f, 100.0f, 0.2f, 1, {0.05f}, {1.0f}, {}, {}, 200, 200},         // Large T
        {100.0f, 100.0f, 1.0f, 10.0f, 1, {0.05f}, {1.0f}, {}, {}, 200, 200},          // Large sigma

        // Invalid/edge: function should return intrinsic or 0
        {0.0f, 100.0f, 1.0f, 0.2f, 1, {0.05f}, {1.0f}, {}, {}, 200, 200},            // S=0
        {100.0f, 0.0f, 1.0f, 0.2f, 0, {0.05f}, {1.0f}, {}, {}, 200, 200},            // K=0
        {100.0f, 100.0f, 0.0f, 0.2f, 1, {0.05f}, {1.0f}, {}, {}, 200, 200},          // T=0
        {100.0f, 100.0f, 1.0f, 0.0f, 1, {0.05f}, {1.0f}, {}, {}, 200, 200},          // sigma=0

        // Extreme dividends and rates
        {100.0f, 100.0f, 1.0f, 0.2f, 1, {10.0f}, {1.0f}, {100.0f}, {0.1f}, 200, 200},  // High rate, large div
        {100.0f, 100.0f, 1.0f, 0.2f, 0, {-1.0f}, {1.0f}, { -1.0f }, {0.5f}, 200, 200}  // Negative rate/div (though unrealistic)
    };

    for (const auto& tc : cases) {
        // Prepare single-element vectors for vectorized call
        std::vector<float> v_spots = {tc.S};
        std::vector<float> v_strikes = {tc.K};
        std::vector<float> v_tenors = {tc.T};
        std::vector<float> v_sigmas = {tc.sigma};
        std::vector<uint8_t> v_rights = {tc.is_call};

        // Call vectorized GPU function
        std::vector<float> gpu_prices = get_v_price_fd_cuda(
            v_spots, v_strikes, v_tenors, v_sigmas, v_rights,
            tc.rates_curve, tc.rates_times,
            tc.div_amounts, tc.div_times,
            tc.time_steps, tc.space_steps
        );

        REQUIRE(gpu_prices.size() == 1);
        float gpu_price = gpu_prices[0];

        // Compute CPU price for comparison
        float cpu_price = price_american_fd_div_host(
            tc.S, tc.K, tc.T, tc.sigma, tc.is_call,
            tc.rates_curve, tc.rates_times,
            tc.div_amounts, tc.div_times,
            tc.time_steps, tc.space_steps
        );

        // Description for dynamic section
        std::string desc = "S=" + std::to_string(tc.S) + ", K=" + std::to_string(tc.K) +
                           ", T=" + std::to_string(tc.T) + ", sigma=" + std::to_string(tc.sigma) +
                           ", is_call=" + std::to_string(tc.is_call);

        DYNAMIC_SECTION(desc) {
            // Robustness checks for GPU
            REQUIRE(std::isfinite(gpu_price));  // Not NaN or Inf
            REQUIRE(gpu_price >= 0.0f);
            REQUIRE(gpu_price <= FLT_MAX);

            // Consistency with CPU (using relative/absolute tolerance)
            const float tolerance = 1e-4f;
            CHECK_THAT(gpu_price, Catch::Matchers::WithinRel(cpu_price, tolerance) ||
                                  Catch::Matchers::WithinAbs(cpu_price, tolerance));
        }
    }
}

TEST_CASE("Consistency: CPU vs GPU FD Implied Volatility", "[iv][cuda]") {
        // 1. Setup Test Grid
        std::vector<uint8_t> rights = {1, 0};
        std::vector<float> tenors = {0.2f, 1.0f};
        std::vector<float> strikes = {90.0f, 100.0f, 110.0f};

        float S = 100.0f;
        float true_sigma = 0.30f; // The "target" volatility we want to recover

        std::vector<float> rates_curve = {0.05f};
        std::vector<float> rates_times = {1.0f};
        std::vector<float> div_amounts = {1.20f};
        std::vector<float> div_times = {0.5f};

        // 2. Generate Target Prices (using CPU pricer as the source of truth)
        std::vector<float> v_spots, v_strikes, v_tenors, v_prices;
        std::vector<uint8_t> v_rights;

        for (auto r : rights) {
            for (auto t : tenors) {
                for (auto k : strikes) {
                    float price = price_american_fd_div_host(S, k, t, true_sigma, r, rates_curve, rates_times, div_amounts, div_times, 200, 200);
                    if (price > 1e-4f) { // Only test options with some value
                        v_spots.push_back(S);
                        v_strikes.push_back(k);
                        v_tenors.push_back(t);
                        v_rights.push_back(r);
                        v_prices.push_back(price);
                    }
                }
            }
        }

        // 3. Run CPU IV Solver
        std::vector<float> cpu_ivs = get_v_iv_fd_cpu(
            v_prices, v_spots, v_strikes, v_tenors, v_rights,
            rates_curve, rates_times, div_amounts, div_times,
            1e-6f, 100, 1e-4f, 5.0f, 200, 200
        );

        // 4. Run GPU IV Solver
        std::vector<float> gpu_ivs = get_v_iv_fd_cuda(
            v_prices, v_spots, v_strikes, v_tenors, v_rights,
            rates_curve, rates_times, div_amounts, div_times,
            1e-6f, 100, 1e-4f, 5.0f, 200, 200
        );

        // 5. Compare results
        REQUIRE(cpu_ivs.size() == v_prices.size());
        REQUIRE(gpu_ivs.size() == v_prices.size());

        const float iv_tolerance = 1e-3f; // IV is more sensitive than price

        for (size_t i = 0; i < cpu_ivs.size(); ++i) {
            std::string type = v_rights[i] ? "Call" : "Put";
            DYNAMIC_SECTION("IV Option " << type << " T=" << v_tenors[i] << " K=" << v_strikes[i]) {
                // Check if they recovered the true sigma
                CHECK_THAT(cpu_ivs[i], Catch::Matchers::WithinAbs(true_sigma, iv_tolerance));
                // Check if GPU matches CPU
                CHECK_THAT(gpu_ivs[i], Catch::Matchers::WithinAbs(cpu_ivs[i], iv_tolerance));
            }
        }
    }

TEST_CASE("Consistency: CPU vs CPU FD Implied Volatility vectorized vs single", "[iv][host]") {
        // 1. Setup Test Grid
        std::vector<uint8_t> rights = {1, 0};
        std::vector<float> tenors = {0.2f, 1.0f};
        std::vector<float> strikes = {90.0f, 100.0f, 110.0f};

        float S = 100.0f;
        float true_sigma = 0.30f; // The "target" volatility we want to recover

        std::vector<float> rates_curve = {0.05f};
        std::vector<float> rates_times = {1.0f};
        std::vector<float> div_amounts = {1.20f};
        std::vector<float> div_times = {0.5f};

        // 2. Generate Target Prices (using CPU pricer as the source of truth)
        std::vector<float> v_spots, v_strikes, v_tenors, v_prices;
        std::vector<uint8_t> v_rights;

        for (auto r : rights) {
            for (auto t : tenors) {
                for (auto k : strikes) {
                    float price = price_american_fd_div_host(S, k, t, true_sigma, r, rates_curve, rates_times, div_amounts, div_times, 200, 200);
                    if (price > 1e-4f) { // Only test options with some value
                        v_spots.push_back(S);
                        v_strikes.push_back(k);
                        v_tenors.push_back(t);
                        v_rights.push_back(r);
                        v_prices.push_back(price);
                    }
                }
            }
        }

        // 3. Run CPU IV Solver
        std::vector<float> cpu_ivs = get_v_iv_fd_cpu(
            v_prices, v_spots, v_strikes, v_tenors, v_rights,
            rates_curve, rates_times, div_amounts, div_times,
            1e-6f, 100, 1e-4f, 5.0f, 200, 200
        );

        // 5. Compare results
        REQUIRE(cpu_ivs.size() == v_prices.size());

        const float iv_tolerance = 1e-3f; // IV is more sensitive than price

        for (size_t i = 0; i < cpu_ivs.size(); ++i) {

            float single_iv = implied_vol_american_fd_host(
                v_prices[i], v_spots[i], v_strikes[i], v_tenors[i], v_rights[i],
                rates_curve, rates_times, div_amounts, div_times,
                1e-6f, 100, 1e-4f, 5.0f, 200, 200
            );

            std::string type = v_rights[i] ? "Call" : "Put";
            DYNAMIC_SECTION("IV Option " << type << " T=" << v_tenors[i] << " K=" << v_strikes[i]) {
                // Check if they recovered the true sigma
                CHECK_THAT(cpu_ivs[i], Catch::Matchers::WithinAbs(true_sigma, iv_tolerance));
                // Check if Single IV matches CPU vectorized IV
                CHECK_THAT(single_iv, Catch::Matchers::WithinAbs(cpu_ivs[i], iv_tolerance));
            }
        }
    }

TEST_CASE("Circular consistency: GPU FD price -> GPU FD IV -> GPU FD price", "[pricing][iv][cuda]") {
        // Goal:
        // 1) Price on GPU with get_v_fd_price_cuda using a known sigma
        // 2) Recover sigma from that GPU price via get_v_iv_fd_cuda_new
        // 3) (Optional but useful) Re-price using recovered sigma and verify we match the original price

        std::vector<uint8_t> rights = {1, 0}; // Call, Put
        std::vector tenors = {0.05f, 0.25f, 1.0f, 3.0f}; // short/medium/long
        float S = 100.0f;

        // strikes cover ITM/ATM/OTM for both calls and puts
        std::vector strikes = {70.0f, 90.0f, 100.0f, 110.0f, 130.0f};

        // Use a couple of vol levels to make sure solver isn't tuned to one case
        std::vector sigmas = {0.15f, 0.30f, 0.60f};

        std::vector rates_curve = {0.03f, 0.1f, 0.03f};
        std::vector rates_times = {0.2f, 0.8f, 1.5f};
        std::vector div_amounts = {1.0f, 1.50f, 2.0f};
        std::vector div_times = {0.3f, 0.9f, 1.3f};

        const int time_steps = 200;
        const int space_steps = 200;

        // Build vectorized inputs
        std::vector<float> v_spots, v_strikes, v_tenors, v_sigmas;
        std::vector<uint8_t> v_rights;

        for (auto r : rights) {
            for (auto t : tenors) {
                for (auto k : strikes) {
                    for (auto sigma : sigmas) {
                        v_spots.push_back(S);
                        v_strikes.push_back(k);
                        v_tenors.push_back(t);
                        v_sigmas.push_back(sigma);
                        v_rights.push_back(r);
                    }
                }
            }
        }

        // 1) GPU price using "true" sigma
        std::vector<float> v_prices = get_v_price_fd_cuda(
            v_spots, v_strikes, v_tenors, v_sigmas, v_rights,
            rates_curve, rates_times,
            div_amounts, div_times,
            time_steps, space_steps
        );
        REQUIRE(v_prices.size() == v_spots.size());

        // Some deep OTM short-dated options can be numerically ~0; filter them out for IV recovery.
        std::vector<float> f_prices, f_spots, f_strikes, f_tenors, f_sigmas;
        std::vector<uint8_t> f_rights;

        f_prices.reserve(v_prices.size());
        f_spots.reserve(v_prices.size());
        f_strikes.reserve(v_prices.size());
        f_tenors.reserve(v_prices.size());
        f_sigmas.reserve(v_prices.size());
        f_rights.reserve(v_prices.size());

        for (size_t i = 0; i < v_prices.size(); ++i) {
            if (std::isfinite(v_prices[i]) && v_prices[i] > 1e-4f) {
                f_prices.push_back(v_prices[i]);
                f_spots.push_back(v_spots[i]);
                f_strikes.push_back(v_strikes[i]);
                f_tenors.push_back(v_tenors[i]);
                f_sigmas.push_back(v_sigmas[i]);
                f_rights.push_back(v_rights[i]);
            }
        }

        REQUIRE(f_prices.size() > 0);

        // 2) Recover IV on GPU from those GPU prices
        std::vector<float> recovered_ivs = get_v_iv_fd_cuda(
            f_prices, f_spots, f_strikes, f_tenors, f_rights,
            rates_curve, rates_times, div_amounts, div_times,
            1e-6f, 100, 1e-4f, 5.0f, time_steps, space_steps
        );

        REQUIRE(recovered_ivs.size() == f_prices.size());

        // 3) Re-price with recovered IV and compare to original price
        std::vector<float> reprices = get_v_price_fd_cuda(
            f_spots, f_strikes, f_tenors, recovered_ivs, f_rights,
            rates_curve, rates_times,
            div_amounts, div_times,
            time_steps, space_steps
        );
        REQUIRE(reprices.size() == f_prices.size());

        const float price_rel_tol = 2e-3f;   // price should close back reasonably well

        for (size_t i = 0; i < f_prices.size(); ++i) {
            const std::string type = f_rights[i] ? "Call" : "Put";
            DYNAMIC_SECTION("Circular " << type
                            << " T=" << f_tenors[i]
                            << " K=" << f_strikes[i]
                            << " sigma=" << f_sigmas[i]) {
                REQUIRE(std::isfinite(recovered_ivs[i]));
                CHECK(recovered_ivs[i] >= 0.0f);

                // IV matches the original sigma. IV can vary a lot and only have sub-cent differences on the price. Not suitable to reconcile

                // Re-priced value matches the original price
                CHECK(std::isfinite(reprices[i]));
                CHECK_THAT(reprices[i], Catch::Matchers::WithinRel(f_prices[i], price_rel_tol) ||
                                       Catch::Matchers::WithinAbs(f_prices[i], 1e-4f));
            }
        }
    }


// TEST_CASE("Circular consistency: CPU FD price -> CPU FD IV -> CPU FD price", "[pricing][iv][cuda]") {
//         // Goal:
//         // 1) Price on GPU with get_v_fd_price_cuda using a known sigma
//         // 2) Recover sigma from that GPU price via get_v_iv_fd_cuda_new
//         // 3) (Optional but useful) Re-price using recovered sigma and verify we match the original price
//
//         std::vector<uint8_t> rights = {1, 0}; // Call, Put
//         std::vector tenors = {0.05f, 0.25f, 1.0f, 3.0f}; // short/medium/long
//         float S = 100.0f;
//
//         // strikes cover ITM/ATM/OTM for both calls and puts
//         std::vector strikes = {70.0f, 90.0f, 100.0f, 110.0f, 130.0f};
//
//         // Use a couple of vol levels to make sure solver isn't tuned to one case
//         std::vector sigmas = {0.15f, 0.30f, 0.60f};
//
//         std::vector rates_curve = {0.03f, 0.1f, 0.03f};
//         std::vector rates_times = {0.2f, 0.8f, 1.5f};
//         std::vector div_amounts = {1.0f, 1.50f, 2.0f};
//         std::vector div_times = {0.3f, 0.9f, 1.3f};
//
//         const int time_steps = 200;
//         const int space_steps = 200;
//
//         // Build vectorized inputs
//         std::vector<float> v_spots, v_strikes, v_tenors, v_sigmas;
//         std::vector<uint8_t> v_rights;
//
//         for (auto r : rights) {
//             for (auto t : tenors) {
//                 for (auto k : strikes) {
//                     for (auto sigma : sigmas) {
//                         v_spots.push_back(S);
//                         v_strikes.push_back(k);
//                         v_tenors.push_back(t);
//                         v_sigmas.push_back(sigma);
//                         v_rights.push_back(r);
//                     }
//                 }
//             }
//         }
//
//         // 1) GPU price using "true" sigma
//         std::vector<float> v_prices = v_fd_price_host(
//             v_spots, v_strikes, v_tenors, v_sigmas, v_rights,
//             rates_curve, rates_times,
//             div_amounts, div_times,
//             time_steps, space_steps
//         );
//         REQUIRE(v_prices.size() == v_spots.size());
//
//         // Some deep OTM short-dated options can be numerically ~0; filter them out for IV recovery.
//         std::vector<float> f_prices, f_spots, f_strikes, f_tenors, f_sigmas;
//         std::vector<uint8_t> f_rights;
//
//         f_prices.reserve(v_prices.size());
//         f_spots.reserve(v_prices.size());
//         f_strikes.reserve(v_prices.size());
//         f_tenors.reserve(v_prices.size());
//         f_sigmas.reserve(v_prices.size());
//         f_rights.reserve(v_prices.size());
//
//         for (size_t i = 0; i < v_prices.size(); ++i) {
//             if (std::isfinite(v_prices[i]) && v_prices[i] > 1e-4f) {
//                 f_prices.push_back(v_prices[i]);
//                 f_spots.push_back(v_spots[i]);
//                 f_strikes.push_back(v_strikes[i]);
//                 f_tenors.push_back(v_tenors[i]);
//                 f_sigmas.push_back(v_sigmas[i]);
//                 f_rights.push_back(v_rights[i]);
//             }
//         }
//
//         REQUIRE(f_prices.size() > 0);
//
//         // 2) Recover IV on GPU from those GPU prices
//         std::vector<float> recovered_ivs = get_v_iv_fd_cpu(
//             f_prices, f_spots, f_strikes, f_tenors, f_rights,
//             rates_curve, rates_times, div_amounts, div_times,
//             1e-6f, 100, 1e-4f, 5.0f, time_steps, space_steps
//         );
//
//         // 3) Re-price with recovered IV and compare to original price
//         std::vector<float> reprices = get_v_fd_price_cuda(
//             f_spots, f_strikes, f_tenors, recovered_ivs, f_rights,
//             rates_curve, rates_times,
//             div_amounts, div_times,
//             time_steps, space_steps
//         );
//         REQUIRE(reprices.size() == f_prices.size());
//
//         const float price_rel_tol = 2e-3f;   // price should close back reasonably well
//
//         for (size_t i = 0; i < f_prices.size(); ++i) {
//             const std::string type = f_rights[i] ? "Call" : "Put";
//             DYNAMIC_SECTION("Circular " << type
//                             << " T=" << f_tenors[i]
//                             << " K=" << f_strikes[i]
//                             << " sigma=" << f_sigmas[i]) {
//                 REQUIRE(std::isfinite(recovered_ivs[i]));
//                 CHECK(recovered_ivs[i] >= 0.0f);
//
//                 // Re-priced value matches original price
//                 CHECK(std::isfinite(reprices[i]));
//                 CHECK_THAT(reprices[i], Catch::Matchers::WithinRel(f_prices[i], price_rel_tol) ||
//                                        Catch::Matchers::WithinAbs(f_prices[i], 1e-4f));
//             }
//         }
//     }