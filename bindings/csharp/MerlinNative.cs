using System;
using System.Runtime.InteropServices;

namespace Merlin
{
    public static class MerlinNative
    {
        // Use "merlin_native" (without .dll extension) for cross-platform compatibility
        private const string DllName = "merlin_native";

        /// <summary>
        /// Compute implied volatilities using a finite difference (binomial tree) pricer.
        /// </summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern float merlin_get_single_iv_cpu(
            float price,
            float spot,
            float strike,
            float tenor,
            [MarshalAs(UnmanagedType.U1)] bool isCall,
            float r,
            [In] float[] divAmounts,
            [In] float[] divTimes,
            int divCount,
            float tol = 0.001f,
            int maxIter = 200,
            float stepsFactor = 1.0f
        );

        // You can also add higher-level helper methods here
        public static float GetImpliedVol(float price, float spot, float strike, float tenor, bool isCall, float r)
        {
            return merlin_get_single_iv_cpu(price, spot, strike, tenor, isCall, r, Array.Empty<float>(), Array.Empty<float>(), 0);
        }

        // <summary>
        /// Compute implied volatilities using a finite difference (binomial tree) pricer.
        /// </summary>
        /// <param name="outIvs">Output array for IVs (must be pre-allocated to 'count' size).</param>
        /// <param name="rights">1 for Call, 0 for Put.</param>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern float merlin_get_single_iv_cuda(
            float price,
            float spot,
            float strike,
            float tenor,
            [MarshalAs(UnmanagedType.U1)] bool isCall,
            float r,
            int nSteps,
            float[] divAmounts,
            float[] divTimes,
            int divCount,
            float tol = 0.001f,
            int maxIter = 200
            );

        /// <summary>
        /// Compute implied volatilities using a finite difference (binomial tree) pricer.
        /// </summary>
        /// <param name="outIvs">Output array for IVs (must be pre-allocated to 'count' size).</param>
        /// <param name="rights">1 for Call, 0 for Put.</param>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void merlin_get_iv_fd(
            float[] outIvs,
            float[] prices,
            float[] spots,
            float[] strikes,
            float[] tenors,
            int[] rights,
            int count,
            float r,
            int nSteps,
            float[] divAmounts,
            float[] divTimes,
            int divCount,
            float tol = 0.01f,
            int max_iter = 200,
            float v_min = 1e-4f,
            float v_max = 5.0f
        );

        /// <summary>
        /// Compute implied volatilities with a time-dependent risk-free rate curve.
        /// </summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void merlin_get_iv_fd_term_structure(
            float[] outIvs,
            float[] prices,
            float[] spots,
            float[] strikes,
            float[] tenors,
            int[] rights,
            int count,
            float[] rCurve,
            float[] timePoints,
            int nCurvePoints,
            int nSteps,
            float[] divAmounts,
            float[] divTimes,
            int divCount
        );

        /// <summary>
        /// Compute American option prices in a vectorized fashion.
        /// </summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void merlin_get_price_fd(
            float[] outPrices,
            float[] spots,
            float[] strikes,
            float[] tenors,
            int[] rights,
            float[] sigmas,
            int count,
            float r,
            int nSteps,
            float[] divAmounts,
            float[] divTimes,
            int divCount
        );

        /// <summary>
        /// Compute American option Deltas.
        /// </summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void merlin_get_delta_fd(
            float[] outDeltas,
            float[] spots,
            float[] strikes,
            float[] tenors,
            int[] rights,
            float[] sigmas,
            int count,
            float r,
            int nSteps,
            float[] divAmounts,
            float[] divTimes,
            int divCount
        );

        /// <summary>
        /// Compute American option Vegas.
        /// </summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void merlin_get_vega_fd(
            float[] outVegas,
            float[] spots,
            float[] strikes,
            float[] tenors,
            int[] rights,
            float[] sigmas,
            int count,
            float r,
            int nSteps,
            float[] divAmounts,
            float[] divTimes,
            int divCount
        );
    }
}