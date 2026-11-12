float[] a = [1f, 2f, 3f, 4f, 5f];
float[] b = [10f, 20f, 30f, 40f, 50f];
float[] result = CalculatorGpu.Add(a, b);
Console.WriteLine(string.Join(", ", result));
