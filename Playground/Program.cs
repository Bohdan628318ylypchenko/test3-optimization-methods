using MathNet.Numerics;
using MathNet.Numerics.Differentiation;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

class Program
{
    static void Main(string[] args)
    {
        //Vector v = (Vector)CreateVector.Dense(new double[] { 3, 4 });
        //Console.WriteLine(v.L2Norm());

        //Func<double[], double> f = x => 3 * x[0] * x[0] + x[0] * x[1] + 2 * x[1] * x[1];
        //double[] point = new double[2] { 6.0, 4.0 };
        //Vector grad = (Vector)CreateVector.Dense(point.Length, i => Differentiate.FirstPartialDerivative(f, point, i));
        //Console.WriteLine(grad);

        //Func<Vector, double> f = v => 3 * v.At(0) * v.At(0) + v.At(0) * v.At(1) + 2 * v.At(1) * v.At(1);
        //Vector point = (Vector)CreateVector.Dense(new double[] { 6.0, 4.0 });
        //Vector result = (Vector)point.MapIndexed<double>((i, x) => Differentiate.FirstPartialDerivative(p => f((Vector)CreateVector.Dense(p)), point.AsArray(), i));
        //Console.WriteLine(result);

        //Func<Vector<double>, double> f = v => 4 * v.At(0) * v.At(0) + v.At(0) * v.At(1) + v.At(1) * v.At(1);
        Func<Vector<double>, double> f = v => 3 * Math.Pow(v.At(0) - 15, 2) - v.At(0) * v.At(1) + 4 * Math.Pow(v.At(1), 2);
        Vector point = (Vector)CreateVector.Dense(new double[] { 14.923, 1.649 });
        Func<double[], double> arrf = p => f(CreateVector.Dense(p));
        double[] arrp = point.AsArray();
        var g = point.MapIndexed<double>((i, x) => Differentiate.FirstPartialDerivative(arrf, arrp, i));
        var s = g.Multiply(-1.0);
        NumericalDerivative nd = new NumericalDerivative();
        double h11 = nd.EvaluateMixedPartialDerivative(arrf, arrp, new int[] { 0, 0 }, 2);
        double h12 = nd.EvaluateMixedPartialDerivative(arrf, arrp, new int[] { 0, 1 }, 2);
        double h21 = nd.EvaluateMixedPartialDerivative(arrf, arrp, new int[] { 1, 0 }, 2);
        double h22 = nd.EvaluateMixedPartialDerivative(arrf, arrp, new int[] { 1, 1 }, 2);
        var H = DenseMatrix.OfArray(new double[,] { { h11, h12 },
                                                    { h21, h22 } });
        //var l = -1.0 * (g * s) / (((s * H) * st)));
        var l = (g * s) / (((s * H) * s));
        Console.WriteLine(l);
    }
}
