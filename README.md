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

## Install

GPUSharp targets .NET 10. [rustc](https://www.rust-lang.org/tools/install) must be on `PATH`.

```bash
dotnet add package GPUSharp --version 0.1.0
```

## Example

From this repository:

```bash
dotnet run --project samples/AddArrays
```

That runs `Calculator.Add` over two 5-element arrays and prints `11, 22, 33, 44, 55`.

## How it works

`src/GPUSharp` defines `[GpuKernel]`. `src/GPUSharp.Generator` finds marked methods at build time and emits the array companion. `rust/gpusharp-compiler` compiles each method into the kernel the companion calls.

## Development

- `samples/AddArrays` — console app that adds two arrays through `CalculatorGpu.Add`.
- `src/GPUSharp` — .NET 10 package and `[GpuKernel]`.
- `src/GPUSharp.Generator` — Roslyn source generator for the array companion.
- `rust/gpusharp-compiler` — kernel compiler. `dll --out <dll> <export-name> <method.cs>` compiles one kernel.

From the repository root:

```bash
dotnet build GPUSharp.slnx
cargo test --manifest-path rust/gpusharp-compiler/Cargo.toml
cargo fmt --manifest-path rust/gpusharp-compiler/Cargo.toml --check
cargo clippy --manifest-path rust/gpusharp-compiler/Cargo.toml --all-targets -- -D warnings
```

`cargo fmt` and `cargo clippy` require the `rustfmt` and `clippy` components (`rustup component add rustfmt clippy`).

## License

GPUSharp is licensed under the [Apache License 2.0](LICENSE).
