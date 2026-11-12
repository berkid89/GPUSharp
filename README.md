# GPUSharp

Write GPU kernels in C# and run them on NVIDIA GPUs.

Mark a scalar method with `[GpuKernel]`. GPUSharp compiles that method and generates an array API next to it. The scalar method remains a normal .NET method. The generated companion runs the kernel over arrays.

```csharp
public static class Calculator
{
    [GpuKernel]
    public static float Add(float a, float b)
    {
        return a + b;
    }
}

float[] result = CalculatorGpu.Add(a, b);
```

## Build status
[![CI](https://github.com/berkid89/GPUSharp/actions/workflows/ci.yml/badge.svg)](https://github.com/berkid89/GPUSharp/actions/workflows/ci.yml)

## Install

GPUSharp targets .NET 10. `dotnet build` works on Windows and Linux. Running a kernel requires Linux, [cargo oxide](https://github.com/NVlabs/cuda-oxide) 0.2.1, the CUDA toolkit, LLVM 21, and an NVIDIA GPU (Ampere, `sm_80`, or newer). Calling a kernel on another operating system throws.

```bash
dotnet add package GPUSharp
```

The published package includes the compiler for Windows and Linux. Windows builds check the kernels and emit the managed API. Linux builds compile the kernel library with `cargo oxide` on `PATH`.

## Example

On a Linux machine with an NVIDIA GPU:

```bash
dotnet run --project samples/AddArrays
```

That runs `Calculator.Add` over two 5-element arrays and prints `11, 22, 33, 44, 55`.

`samples/AddArrays.NuGet` is the same program with a floating `PackageReference` (`Version="*"`) so restore picks up the latest GPUSharp on nuget.org. Its `nuget.config` lists only nuget.org, so the build does not use a local package source or the projects in this repo:

```bash
dotnet run --project samples/AddArrays.NuGet
```

## How it works

`src/GPUSharp` defines `[GpuKernel]`. `src/GPUSharp.Generator` finds marked methods at build time and emits the array companion. `rust/gpusharp-compiler` compiles each method into a cuda-oxide kernel and a host library that launches it.

## Development

- `samples/AddArrays` — console app that adds two arrays through `CalculatorGpu.Add`.
- `samples/AddArrays.NuGet` — the same app, restored from the GPUSharp package on nuget.org.
- `src/GPUSharp` — .NET 10 package and `[GpuKernel]`.
- `src/GPUSharp.Generator` — Roslyn source generator for the array companion.
- `rust/gpusharp-compiler` — kernel compiler. `dll --out <so> <export-name> <method.cs>` compiles one kernel with `cargo oxide`.

`cargo test` checks the emitted kernel. It does not need a GPU or the CUDA toolkit. `dotnet build` compiles the sample kernel and needs the Linux toolchain above. `dotnet run` launches it and needs a GPU.

From the repository root:

```bash
cargo test --manifest-path rust/gpusharp-compiler/Cargo.toml
cargo fmt --manifest-path rust/gpusharp-compiler/Cargo.toml --check
cargo clippy --manifest-path rust/gpusharp-compiler/Cargo.toml --all-targets -- -D warnings
dotnet build GPUSharp.slnx
```

`cargo fmt` and `cargo clippy` require the `rustfmt` and `clippy` components (`rustup component add rustfmt clippy`). `GPUSHARP_CUDA_ARCH` overrides the default `sm_80` target.

## License

GPUSharp is licensed under the [Apache License 2.0](LICENSE).

## Contributing

> This is a young project, and a good time to help shape it. Ideas, bug reports, and pull requests are all welcome!
