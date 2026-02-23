#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "../../src/core/pricing_engine.h"
#include "../../src/core/calibration/ssvi.h"

namespace py = pybind11;

PYBIND11_MODULE(merlin, m) {
    m.doc() = "CUDA-accelerated American option implied volatility calculator";

    m.def("get_v_iv_fd_cpu", &get_v_iv_fd_cpu,
          "Compute implied volatilities with yield curve and single dividend schedule",
          py::arg("prices"),
          py::arg("spots"),
          py::arg("strikes"),
          py::arg("tenors"),
          py::arg("v_is_call"),
          py::arg("rates_curve") = std::vector<float>(),
          py::arg("rates_times") = std::vector<float>(),
          py::arg("div_amounts") = std::vector<float>(),
          py::arg("div_times") = std::vector<float>(),
          py::arg("tol") = 1e-6f,
          py::arg("max_iter") = 100,
          py::arg("v_min") = 1e-4f,
          py::arg("v_max") = 5.0f,
          py::arg("time_steps") = 200,
          py::arg("space_steps") = 200
          );

    m.def("get_v_iv_fd_gpu", &get_v_iv_fd_cuda,
          "Compute implied volatilities with yield curve and single dividend schedule",
          py::arg("prices"),
          py::arg("spots"),
          py::arg("strikes"),
          py::arg("tenors"),
          py::arg("v_is_call"),
          py::arg("rates_curve") = std::vector<float>(),
          py::arg("rates_times") = std::vector<float>(),
          py::arg("div_amounts") = std::vector<float>(),
          py::arg("div_times") = std::vector<float>(),
          py::arg("tol") = 1e-6f,
          py::arg("max_iter") = 100,
          py::arg("v_min") = 1e-4f,
          py::arg("v_max") = 5.0f,
          py::arg("time_steps") = 200,
          py::arg("space_steps") = 200
          );

    m.def("get_v_fd_price", &get_v_price_fd_cuda,
        "Compute American option prices in vectorized fashion with per-option sigma and yield curve",
        py::arg("spots"), py::arg("strikes"), py::arg("tenors"),
        py::arg("sigmas"), py::arg("v_is_call"),
        py::arg("rates_curve") = std::vector<float>(), py::arg("rates_times") = std::vector<float>(),
        py::arg("div_amounts") = std::vector<float>(), py::arg("div_times") = std::vector<float>(),
        py::arg("time_steps"), py::arg("space_steps")
        );

    m.def("get_fd_price_cpu", &price_american_fd_div_host,
        "Compute American option prices in vectorized fashion with per-option sigma and yield curve",
        py::arg("s"), py::arg("k"), py::arg("t"),
        py::arg("sigma"), py::arg("is_call"),
        py::arg("rates_curve") = std::vector<float>(), py::arg("rates_times") = std::vector<float>(),
        py::arg("div_amounts") = std::vector<float>(), py::arg("div_times") = std::vector<float>(),
        py::arg("time_steps"), py::arg("space_steps")
        );

    m.def("get_v_fd_price_cpu", &v_fd_price_host,
        "Compute American option prices in vectorized fashion with per-option sigma and yield curve",
        py::arg("spots"), py::arg("strikes"), py::arg("tenors"),
        py::arg("sigmas"), py::arg("v_is_call"),
        py::arg("rates_curve") = std::vector<float>(), py::arg("rates_times") = std::vector<float>(),
        py::arg("div_amounts") = std::vector<float>(), py::arg("div_times") = std::vector<float>(),
        py::arg("time_steps"), py::arg("space_steps")
        );

    // Calibration residuals: E-SSVI -> IV -> Price -> residuals
    m.def("f_min_price_surface_theta_rho_psi",
        &f_min_price_surface_theta_rho_psi_cuda,
        "Build E-SSVI IVs from tenor params, price with CUDA, and return residuals (model - market)",
        py::arg("calibration_params"),
        py::arg("item_prices"),
        py::arg("s"),
        py::arg("k"),
        py::arg("t"),
        py::arg("v_is_call"),
        py::arg("mny_fwd_ln"),
        py::arg("tenor_index"),
        py::arg("n_steps") = 200,
        py::arg("rates_curve") = std::vector<float>(),
        py::arg("rates_times") = std::vector<float>(),
        py::arg("div_amounts") = std::vector<float>(),
        py::arg("div_times") = std::vector<float>());

}