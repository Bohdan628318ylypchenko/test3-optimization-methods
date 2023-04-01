using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;

namespace FastestSearch
{
    public abstract class FSFunc2Minimizer
    {
        public Vector<double>[] MinimizeFunc(Func<Vector<double>, double> f,
                                             Vector<double> startPoint,
                                             double e)
        {
            // List to store path
            LinkedList<Vector<double>> result = new LinkedList<Vector<double>>();
            result.AddLast(startPoint);

            // Minimizing
            Vector<double> currentPoint = startPoint;
            while(!StopCriteria(f, currentPoint, e))
            {
                currentPoint = currentPoint - CalcGrad(f, currentPoint) * CalcLambda(f, currentPoint);
                result.AddLast(currentPoint);
            }
            
            // Returning
            return result.ToArray();
        }      

        private bool StopCriteria(Func<Vector<double>, double> f,
                                  Vector<double> point,
                                  double e)
        {
            // Gradient module less than e
            var module = CalcGrad(f, point).L2Norm();
            return module <= e;
        }

        protected Vector<double> CalcGrad(Func<Vector<double>, double> f,
                                          Vector<double> point)
        {
            // Map copy 
            return point.MapIndexed<double>((i, x) => Differentiate.FirstPartialDerivative(p => f(CreateVector.Dense(p)), point.AsArray(), i));
        }

        protected abstract double CalcLambda(Func<Vector<double>, double> f, Vector<double> point);
    }
}