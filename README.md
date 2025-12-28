# option-pricer
CUDA enabled option pricer with python stubs to price millions of independent american equity options with fixed cash dividends.

## Installation

### Python

1. Start → “x64 Native Tools Command Prompt for VS 2022” (or whatever VS version you use).
2. Run the following commands in the command prompt:
   ```sh
   python -m pip install .\bindings\python
   ```
   

### C# / .NET
```sh
  cmake --build build\bin\Release --config Release
  cmake --install cmake-build-debug-visual-studio --config Debug --prefix ./dist_native
  cd .\bindigs\csharp
  dotnet build -c Release
```