#pragma once

// Platform-specific symbol visibility macros
#ifdef _WIN32
    #ifdef MERLIN_EXPORTS
        #define API __declspec(dllexport)
    #else
        #define API __declspec(dllimport)
    #endif
#else
    #define API __attribute__((visibility("default")))
#endif
