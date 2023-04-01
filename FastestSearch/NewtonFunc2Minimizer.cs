using MathNet.Numerics.Differentiation;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace Func2Minimizers
{
    public class NewtonFunc2Minimizer : Func2Minimizer
    {
        protected override Vector<double> CalcNextPoint(Vector<double> currentPoint, 
                                                        Func<Vector<double>, double> f)
        {
            return currentPoint - CalcH(f, currentPoint).Inverse() * CalcGrad(f, currentPoint);
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
