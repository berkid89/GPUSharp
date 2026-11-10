//! Compiler component for GPUSharp.
//!
//! This library will eventually consume the published `csharp-to-rust` crate,
//! inspect its typed intermediate representation, and generate GPU-oriented code
//! for later compilation to PTX. Parsing, kernel generation, and GPU execution
//! are not implemented yet.
