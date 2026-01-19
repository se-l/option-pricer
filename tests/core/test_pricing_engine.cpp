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

    std::vector rates_curve = {0.04f, 0.05f};
    std::vector rates_times = {0.5f, 1.0f};
    std::vector div_amounts = {1.0f, 1.50f, 2.0f};
    std::vector div_times = {0.4f, 0.8f, 1.2f};

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
    std::vector<float> gpu_results = get_v_fd_price_cuda(
        v_spots, v_strikes, v_tenors, v_sigmas, v_rights,
        rates_curve, rates_times,
        div_amounts, div_times,
        200, 200
    );

    // Verify GPU returned the expected number of results
    REQUIRE(gpu_results.size() == v_spots.size());

    // 4. Compare with CPU Results
    const float tolerance = 1e-4f;

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
        std::vector<float> gpu_ivs = get_v_iv_fd_cuda_new(
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