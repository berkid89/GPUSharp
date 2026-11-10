# GPUSharp

Experimental C#/.NET GPU computing framework with a Rust-based compilation component.

> **Work in progress / early prototype.** GPU execution and `[GpuKernel]` are not implemented yet. The repository currently contains a buildable scaffold only.

## Vision

GPUSharp is intended to let .NET developers write GPU-oriented computation in C# and run it on NVIDIA GPUs without hand-writing CUDA kernels.

A scalar C# method would describe the per-element calculation. GPUSharp would turn that method into an element-wise GPU kernel, transfer array data, run the kernel, and expose a convenient .NET API. The generated companion API is the long-term developer experience. Placing `[GpuKernel]` on an existing method will not intercept ordinary method calls.

**Future API — not yet implemented:**

```csharp
public static class Calculator
{
    [GpuKernel]
    public static float Add(float a, float b) => a + b;
}

float[] result = CalculatorGpu.Add(a, b);
```

## Architecture

Three layers are planned, and none of them are wired up in this scaffold. The .NET side will own the public API, the `[GpuKernel]` attribute, a Roslyn source generator, and GPU runtime orchestration. The Rust crate `gpusharp-compiler` will turn supported C# methods into GPU-oriented code compiled to PTX. An NVIDIA runtime will load the compiled kernels, manage device memory, and return results to .NET. The Rust CUDA toolchain and the .NET–Rust integration mechanism are left for later milestones.

## Prerequisites

- [.NET 10 SDK](https://dotnet.microsoft.com/download/dotnet/10.0)
- [Rust and Cargo](https://www.rust-lang.org/tools/install) (stable Rust, edition 2024)
- A Git client

**No NVIDIA GPU or CUDA toolkit is required for this initial scaffold.** Later GPU milestones will require a suitable NVIDIA GPU environment.

## Build and run

From the repository root:

```bash
dotnet build GPUSharp.slnx
cargo build --manifest-path rust/gpusharp-compiler/Cargo.toml
cargo run --manifest-path rust/gpusharp-compiler/Cargo.toml
```

The Rust CLI prints a placeholder message. It does not compile kernels.

## Development

- `src/GPUSharp` — .NET 10 class library and the future public API. It has no kernels, attributes, or native interop yet.
- `rust/gpusharp-compiler` — Cargo package (Rust import name `gpusharp_compiler`) with a library entry point and a minimal CLI. This is the future compiler and code-generation component.

Formatting and checks, from the repository root:

```bash
dotnet build GPUSharp.slnx
cargo test --manifest-path rust/gpusharp-compiler/Cargo.toml
cargo fmt --manifest-path rust/gpusharp-compiler/Cargo.toml --check
cargo clippy --manifest-path rust/gpusharp-compiler/Cargo.toml --all-targets -- -D warnings
```

`cargo fmt` and `cargo clippy` require the `rustfmt` and `clippy` components (`rustup component add rustfmt clippy`).

## Roadmap

Repository scaffolding:

- [x] **Milestone 0 — Repository scaffold:** .NET 10 class library and Rust compiler package build independently, with no GPU or CUDA toolchain.

GPU milestones:

- [ ] **Milestone 1 — Rust GPU proof of concept:** compile and execute a manually written element-wise vector-add kernel on NVIDIA hardware.
- [ ] **Milestone 2 — Transpiler integration:** use a C# scalar method as the calculation in a generated kernel.
- [ ] **Milestone 3 — .NET execution:** transfer `float[]` buffers, launch the GPU kernel, and return correct results to .NET.
- [ ] **Milestone 4 — C# developer experience:** introduce `[GpuKernel]` and a Roslyn source generator producing typed wrappers such as `CalculatorGpu.Add(...)`.
- [ ] **Later:** broader type support, diagnostics, memory/performance optimization, and more GPU operations.

## License

GPUSharp is licensed under the [Apache License 2.0](LICENSE).
