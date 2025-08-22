#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "../core/option_pricer.h"
#include "../core/calibration/ssvi.h"

namespace py = pybind11;

PYBIND11_MODULE(berlin, m) {
    m.doc() = "CUDA-accelerated American option implied volatility calculator";

    // New simplified interface with single dividend schedule
    m.def("get_v_iv_fd_single_underlying", &get_v_iv_fd_cuda,
          "Compute implied volatilities with single dividend schedule for all options",
          py::arg("prices"), py::arg("spots"), py::arg("strikes"), py::arg("tenors"), py::arg("rights"),
          py::arg("r"), py::arg("n_steps") = 100,
          py::arg("div_amounts") = std::vector<float>(),
          py::arg("div_times") = std::vector<float>());

    // Keep existing functions for compatibility
    m.def("get_v_iv_fd_with_term_structure", &get_v_iv_fd_with_term_structure_cuda,
          "Compute implied volatilities using finite difference pricer with time-dependent risk-free rate (slower)",
          py::arg("prices"), py::arg("spots"), py::arg("strikes"), py::arg("tenors"), py::arg("rights"),
          py::arg("r_curve"), py::arg("time_points"), py::arg("n_steps") = 100,
          py::arg("div_amounts") = std::vector<float>(),
          py::arg("div_times") = std::vector<float>());

    m.def("get_v_fd_price", &get_v_fd_price_cuda,
    "Compute American option prices in vectorized fashion with per-option sigma",
    py::arg("spots"), py::arg("strikes"), py::arg("tenors"), py::arg("rights"),
    py::arg("sigmas"), py::arg("r"), py::arg("n_steps") = 100,
    py::arg("div_amounts") = std::vector<float>(), py::arg("div_times") = std::vector<float>());

    m.def("get_v_fd_delta", &get_v_fd_delta_cuda,
        "Compute American option deltas in vectorized fashion with per-option sigma",
        py::arg("spots"), py::arg("strikes"), py::arg("tenors"), py::arg("rights"),
        py::arg("sigmas"), py::arg("r"), py::arg("n_steps") = 100,
        py::arg("div_amounts") = std::vector<float>(), py::arg("div_times") = std::vector<float>());

    m.def("get_v_fd_vega", &get_v_fd_vega_cuda,
        "Compute American option vegas in vectorized fashion with per-option sigma",
        py::arg("spots"), py::arg("strikes"), py::arg("tenors"), py::arg("rights"),
        py::arg("sigmas"), py::arg("r"), py::arg("n_steps") = 100,
        py::arg("div_amounts") = std::vector<float>(), py::arg("div_times") = std::vector<float>());

    // Calibration residuals: E-SSVI -> IV -> Price -> residuals
    m.def("f_min_price_surface_theta_rho_psi",
        &f_min_price_surface_theta_rho_psi_cuda,
        "Build E-SSVI IVs from tenor params, price with CUDA, and return residuals (model - market)",
        py::arg("calibration_params"),
        py::arg("item_prices"),
        py::arg("s"),
        py::arg("k"),
        py::arg("t"),
        py::arg("rights"),
        py::arg("mny_fwd_ln"),
        py::arg("tenor_index"),
        py::arg("r"),
        py::arg("n_steps") = 100,
        py::arg("div_amounts") = std::vector<float>(),
        py::arg("div_times") = std::vector<float>());

}