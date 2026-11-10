//! Adapter over the published `csharp-to-rust` crate.
//!
//! GPUSharp does not parse C#. [`analyze_and_emit`] asks that crate to check a
//! method and emit the equivalent Rust function. This does not generate or run
//! a GPU kernel.

/// Check one C# method and return the Rust function produced by `csharp-to-rust`.
///
/// Parsing, type checking, and Rust emission stay in the upstream crate.
///
/// # Errors
///
/// Returns the upstream [`csharp_to_rust::Error`] unchanged when the source is
/// invalid or outside the supported subset.
pub fn analyze_and_emit(source: &str) -> Result<String, csharp_to_rust::Error> {
    let method = csharp_to_rust::analyze_method(source)?;
    Ok(csharp_to_rust::emit_rust(&method))
}
