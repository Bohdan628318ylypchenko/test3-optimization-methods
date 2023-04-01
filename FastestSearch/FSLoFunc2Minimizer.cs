using MathNet.Numerics.Differentiation;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace FastestSearch
{
    public class FSLoFunc2Minimizer : FSFunc2Minimizer
    {
        protected override double CalcLambda(Func<Vector<double>, double> f, Vector<double> point)
        {
            var g = CalcGrad(f, point);
            var s = g.Multiply(-1.0);
            
            var H = CalcH(f, point);

            return -1.0 * (g * s) / (((s * H) * s));
        }

        private Matrix<double> CalcH(Func<Vector<double>, double> f, Vector<double> point)
        {
            // double[] wrappers
            Func<double[], double> arrf = p => f(CreateVector.Dense(p));
            double[] arrp = point.AsArray();

            // Derivatives
            NumericalDerivative nd = new NumericalDerivative();
            double h11 = nd.EvaluateMixedPartialDerivative(arrf, arrp, new int[] { 0, 0 }, 2);
            double h12 = nd.EvaluateMixedPartialDerivative(arrf, arrp, new int[] { 0, 1 }, 2);
            double h21 = nd.EvaluateMixedPartialDerivative(arrf, arrp, new int[] { 1, 0 }, 2);
            double h22 = nd.EvaluateMixedPartialDerivative(arrf, arrp, new int[] { 1, 1 }, 2);

            // Returning
            return DenseMatrix.OfArray(new double[,] { { h11, h12 }, 
                                                       { h21, h22 } });
        }
    }
}
