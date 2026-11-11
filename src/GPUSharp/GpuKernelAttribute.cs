namespace GPUSharp;

/// Marks a scalar method. A source generator adds an element-wise array companion.
[AttributeUsage(AttributeTargets.Method, Inherited = false, AllowMultiple = false)]
public sealed class GpuKernelAttribute : Attribute;
