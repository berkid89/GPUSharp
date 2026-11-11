# GPUSharp

Experimental C#/.NET GPU computing framework with a Rust-based compilation component.

> **Work in progress / early prototype.** `[GpuKernel]` can compile a scalar `float` method into a CPU array companion. GPU execution is not implemented yet.

## Vision

GPUSharp is intended to let .NET developers write GPU-oriented computation in C# and run it on NVIDIA GPUs without hand-writing CUDA kernels.

A scalar C# method describes the per-element calculation. GPUSharp turns that method into an element-wise kernel and exposes a generated companion API. Placing `[GpuKernel]` on a method does not intercept ordinary method calls.

The sample runs that companion on the CPU:

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

## Architecture

The sample wires three pieces. `src/GPUSharp` defines `[GpuKernel]`. `src/GPUSharp.Generator` finds marked methods at `dotnet build`, asks `gpusharp-compiler` to compile them, and emits `CalculatorGpu`. The Rust crate compiles the scalar method to an element-wise host function and a CPU dll. CUDA, PTX, and device memory are not implemented.

## Prerequisites

- [.NET 10 SDK](https://dotnet.microsoft.com/download/dotnet/10.0)
- [Rust and Cargo](https://www.rust-lang.org/tools/install) (stable Rust, edition 2024), including `rustc`
- A Git client

**No NVIDIA GPU or CUDA toolkit is required.** The sample runs the generated kernel on the CPU.

## Build and run

From the repository root:

```bash
dotnet run --project samples/AddArrays
```

That builds the host dll from `Calculator.Add` and prints `11, 22, 33, 44, 55`.

`cargo run --manifest-path rust/gpusharp-compiler/Cargo.toml` still prints the Rust source for a hardcoded `Add` sample. It does not run that sample.

## Development

- `samples/AddArrays` — console caller. It adds two 5-element arrays through `CalculatorGpu.Add`.
- `src/GPUSharp` — .NET 10 library. It contains `[GpuKernel]`.
- `src/GPUSharp.Generator` — Roslyn source generator. It builds `gpusharp_kernels` during compilation and emits the array companion.
- `rust/gpusharp-compiler` — Cargo package (Rust import name `gpusharp_compiler`). `dll --out <dll> <export-name> <method.cs>` compiles one kernel into a host cdylib.

Formatting and checks, from the repository root:

```bash
dotnet build GPUSharp.slnx
cargo test --manifest-path rust/gpusharp-compiler/Cargo.toml
cargo fmt --manifest-path rust/gpusharp-compiler/Cargo.toml --check
cargo clippy --manifest-path rust/gpusharp-compiler/Cargo.toml --all-targets -- -D warnings
```

`cargo fmt` and `cargo clippy` require the `rustfmt` and `clippy` components (`rustup component add rustfmt clippy`).

## License

GPUSharp is licensed under the [Apache License 2.0](LICENSE).
