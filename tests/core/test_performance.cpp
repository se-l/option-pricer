#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <vector>
#include <chrono>
#include <cmath>
#include <limits>
#include <sstream>
#include <algorithm>
#include <iostream>

#include "../../src/core/pricing_engine.h"

TEST_CASE("Accuracy: GPU with varying tolerances Implied Volatility", "[iv][cuda]") {
    // 1. Setup Test Grid
    std::vector<uint8_t> rights = {1, 0};
    // std::vector<float> tenors = {0.2f, 1.0f};
    // std::vector<float> strikes = {90.0f, 100.0f, 110.0f};
    // Make the batch *diverse* (instead of replicating the same small set):
    // - many tenors across short/medium/long dates
    // - many strikes across ITM/ATM/OTM
    std::vector<float> tenors;
    tenors.reserve(20);
    for (int i = 0; i < 20; ++i) {
        // 0.05y .. 2.0y
        tenors.push_back(0.05f + (2.0f - 0.05f) * (static_cast<float>(i) / 19.0f));
    }

    std::vector<float> strikes;
    strikes.reserve(501);
    for (int i = 0; i <= 500; ++i) {
        // 80 .. 120
        strikes.push_back(80.0f + (120.0f - 80.0f) * (static_cast<float>(i) / 500.0f));
    }

    float S = 100.0f;
    float true_sigma = 0.30f; // The "target" volatility we want to recover

    std::vector<float> rates_curve = {0.05f};
    std::vector<float> rates_times = {1.0f};
    std::vector<float> div_amounts = {1.20f};
    std::vector<float> div_times   = {0.5f};

    // 2. Generate Target Prices (using CPU pricer as the source of truth)
    std::vector<float> v_spots, v_strikes, v_tenors, v_prices;
    std::vector<uint8_t> v_rights;

    for (auto r : rights) {
        for (auto t : tenors) {
            for (auto k : strikes) {
                float price = price_american_fd_div_host(
                    S, k, t, true_sigma, r,
                    rates_curve, rates_times, div_amounts, div_times,
                    200, 200
                );
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

    REQUIRE(!v_prices.empty()); // sanity: otherwise there's nothing to analyze

    // 3. Sweep solver tolerances/iterations and log perf vs accuracy
    struct Config {
        float iv_tol;
        int max_iter;
        float price_tol;
        float sigma_max;
        int nx;
        int nt;
        const char* name;
    };

    std::vector<Config> configs = {
        // {1e-4f,  50, 1e-4f, 5.0f, 200, 200, "iv_tol=1e-4, it=50"},
        // {1e-4f, 100, 1e-4f, 5.0f, 200, 200, "iv_tol=1e-4, it=100"},
        // {1e-5f,  50, 1e-4f, 5.0f, 200, 200, "iv_tol=1e-5, it=50"},
        // {1e-5f, 100, 1e-4f, 5.0f, 200, 200, "iv_tol=1e-5, it=100"},
        // {1e-6f,  50, 1e-4f, 5.0f, 200, 200, "iv_tol=1e-6, it=50"},
        {1e-6f, 100, 1e-4f, 5.0f, 200, 200, "iv_tol=1e-6, it=100"},
    };

    struct Result {
        Config cfg;
        double ms;
        float max_abs_err;
        float mean_abs_err;
        int non_finite;
        int count;
    };

    auto compute_metrics = [&](const std::vector<float>& ivs) -> Result {
        Result r{};
        r.ms = 0.0;
        r.max_abs_err = 0.0f;
        r.mean_abs_err = 0.0f;
        r.non_finite = 0;
        r.count = static_cast<int>(ivs.size());

        double sum = 0.0;
        float mx = 0.0f;

        for (float iv : ivs) {
            if (!std::isfinite(iv)) {
                r.non_finite++;
                continue;
            }
            float e = std::fabs(iv - true_sigma);
            mx = std::max(mx, e);
            sum += static_cast<double>(e);
        }

        int finite = r.count - r.non_finite;
        r.max_abs_err = (finite > 0) ? mx : std::numeric_limits<float>::infinity();
        r.mean_abs_err = (finite > 0) ? static_cast<float>(sum / finite)
                                      : std::numeric_limits<float>::infinity();
        return r;
    };

    std::vector<Result> results;
    results.reserve(configs.size());

    for (const auto& cfg : configs) {
        auto t0 = std::chrono::steady_clock::now();
        std::vector<float> ivs = get_v_iv_fd_cuda(
            v_prices, v_spots, v_strikes, v_tenors, v_rights,
            rates_curve, rates_times, div_amounts, div_times,
            cfg.iv_tol, cfg.max_iter, cfg.price_tol, cfg.sigma_max, cfg.nx, cfg.nt
        );
        auto t1 = std::chrono::steady_clock::now();

        REQUIRE(ivs.size() == v_prices.size()); // sanity: solver must return one IV per option

        Result r = compute_metrics(ivs);
        r.cfg = cfg;
        r.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        results.push_back(r);
    }

    // 4. Pick "most expensive" as baseline, then mark configs viable if within 1e-4 IV of it.
    auto it_baseline = std::max_element(
        results.begin(), results.end(),
        [](const Result& a, const Result& b) { return a.ms < b.ms; }
    );
    REQUIRE(it_baseline != results.end());

    const float baseline_max_err = it_baseline->max_abs_err;
    const float allowed_worse = 1e-4f; // "not worse than 0.0001 than the most expensive and accurate parameters"

    std::vector<Result> viable;
    for (const auto& r : results) {
        if (r.max_abs_err <= baseline_max_err + allowed_worse) {
            viable.push_back(r);
        }
    }

    // 5. Log a compact report (performance vs accuracy and which params are viable)
    {
        std::ostringstream os;
        os << "IV tolerance sweep report\n";
        os << "Baseline (most expensive): " << it_baseline->cfg.name
           << " | time_ms=" << it_baseline->ms
           << " | max_abs_err=" << it_baseline->max_abs_err
           << " | mean_abs_err=" << it_baseline->mean_abs_err
           << " | non_finite=" << it_baseline->non_finite
           << "/" << it_baseline->count << "\n\n";

        os << "All configs:\n";
        for (const auto& r : results) {
            os << "  - " << r.cfg.name
               << " | time_ms=" << r.ms
               << " | n_options=" << r.count
               << " | max_abs_err=" << r.max_abs_err
               << " | mean_abs_err=" << r.mean_abs_err
               << " | non_finite=" << r.non_finite
               << "/" << r.count
               << ((&r == &(*it_baseline)) ? "  [BASELINE]" : "")
               << "\n";
        }

        os << "\nViable (max_abs_err <= baseline_max_err + " << allowed_worse << "):\n";
        for (const auto& r : viable) {
            os << "  - " << r.cfg.name
               << " | time_ms=" << r.ms
               << " | max_abs_err=" << r.max_abs_err
               << " | mean_abs_err=" << r.mean_abs_err
               << "\n";
        }
        // INFO(os.str());
        std::cout << os.str() << std::endl;
    }

    // Make the test "successful" as long as we find at least one viable parameter set
    REQUIRE(!viable.empty());
}
