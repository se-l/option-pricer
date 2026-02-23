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
        public static extern float merlin_implied_vol_american_fd_host(
            float price,
            float spot,
            float strike,
            float tenor,
            [MarshalAs(UnmanagedType.U1)] bool isCall,
            [In] float[] ratesCurve,
            [In] float[] ratesTimes,
            int ratesCount,
            [In] float[] divAmounts,
            [In] float[] divTimes,
            int divCount,
            float tol         = 1e-6f,
            int   max_iter    = 100,
            float v_min       = 1e-4f,
            float v_max       = 5.0f,
            int   time_steps  = 200,
            int   space_steps = 200
        );

        /// <summary>
        /// Compute vectorized implied volatilities using CPU.
        /// </summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void merlin_get_iv_fd_cpu(
            [Out] float[] outIvs,
            [In] float[] prices,
            [In] float[] spots,
            [In] float[] strikes,
            [In] float[] tenors,
            [In] byte[] vIsCall, // Use byte[] for uint8_t* compatibility
            int count,
            [In] float[] ratesCurve,
            [In] float[] ratesTimes,
            int ratesCount,
            [In] float[] divAmounts,
            [In] float[] divTimes,
            int divCount,
            float tol = 1e-6f,
            int maxIter = 100,
            float vMin = 1e-4f,
            float vMax = 5.0f,
            int time_steps = 200,
            int space_steps = 200
        );

        /// <summary>
        /// Compute vectorized implied volatilities using CUDA.
        /// </summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void merlin_get_iv_fd_gpu(
            [Out] float[] outIvs,
            [In] float[] prices,
            [In] float[] spots,
            [In] float[] strikes,
            [In] float[] tenors,
            [In] byte[] vIsCall, // Use byte[] for uint8_t* compatibility
            int count,
            [In] float[] ratesCurve,
            [In] float[] ratesTimes,
            int ratesCount,
            [In] float[] divAmounts,
            [In] float[] divTimes,
            int divCount,
            float tol = 1e-6f,
            int maxIter = 100,
            float vMin = 1e-4f,
            float vMax = 5.0f,
            int time_steps = 200,
            int space_steps = 200
        );

        /// <summary>
        /// Price a single American option with dividends using CPU FD.
        /// </summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern float merlin_get_price_fd_cpu(
            float spot,
            float strike,
            float tenor,
            float sigma,
            [MarshalAs(UnmanagedType.U1)] bool isCall,
            [In] float[] ratesCurve,
            [In] float[] ratesTimes,
            int ratesCount,
            [In] float[] divAmounts,
            [In] float[] divTimes,
            int divCount,
            int timeSteps = 200,
            int spaceSteps = 200
        );

        /// <summary>
        /// Compute American option prices in a vectorized fashion using CUDA.
        /// </summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void merlin_get_price_fd_cuda(
            [Out] float[] outPrices,
            [In] float[] spots,
            [In] float[] strikes,
            [In] float[] tenors,
            [In] byte[] vIsCall,
            [In] float[] sigmas,
            int count,
            [In] float[] ratesCurve,
            [In] float[] ratesTimes,
            int ratesCount,
            [In] float[] divAmounts,
            [In] float[] divTimes,
            int divCount,
            int timeSteps = 200,
            int spaceSteps = 200
        );

        /// <summary>
        /// Compute vectorized FD delta using CUDA.
        /// </summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void merlin_get_delta_fd_cuda(
            [Out] float[] outDeltas,
            [In] float[] spots,
            [In] float[] strikes,
            [In] float[] tenors,
            [In] byte[] vIsCall,
            [In] float[] sigmas,
            int count,
            int nSteps,
            [In] float[] ratesCurve,
            [In] float[] ratesTimes,
            int ratesCount,
            [In] float[] divAmounts,
            [In] float[] divTimes,
            int divCount
        );

        /// <summary>
        /// Compute vectorized FD vega using CUDA.
        /// </summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void merlin_get_vega_fd_cuda(
            [Out] float[] outVegas,
            [In] float[] spots,
            [In] float[] strikes,
            [In] float[] tenors,
            [In] byte[] vIsCall,
            [In] float[] sigmas,
            int count,
            int nSteps,
            [In] float[] ratesCurve,
            [In] float[] ratesTimes,
            int ratesCount,
            [In] float[] divAmounts,
            [In] float[] divTimes,
            int divCount
        );
    }
}