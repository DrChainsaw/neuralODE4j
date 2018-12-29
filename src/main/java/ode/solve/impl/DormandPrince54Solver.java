package ode.solve.impl;

import ode.solve.api.FirstOrderEquation;
import ode.solve.api.FirstOrderSolver;
import ode.solve.impl.util.ButcherTableu;
import ode.solve.impl.util.SolverConfig;
import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.nd4j.linalg.ops.transforms.Transforms.*;

/**
 * Implementation of the Dormand-Prince method for solving ordinary differential equations.
 * <br><br>
 * Translated implementation from {@link org.apache.commons.math3.ode.nonstiff.DormandPrince54Integrator} to use INDArrays.
 *
 * @author Christian Skarby
 */
public class DormandPrince54Solver implements FirstOrderSolver {

    private static final ButcherTableu butcherTableu =
            new ButcherTableu(
                    // a
                    new INDArray[]{
                            Nd4j.create(new double[]
                                    {1.0 / 5.0}),
                            Nd4j.create(new double[]
                                    {3.0 / 40.0, 9.0 / 40.0}),
                            Nd4j.create(new double[]
                                    {44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0}),
                            Nd4j.create(new double[]
                                    {19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0}),
                            Nd4j.create(new double[]
                                    {9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0}),
                            Nd4j.create(new double[]
                                    {35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0})
                    },
                    // first b
                    Nd4j.create(new double[]{
                            35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0.0
                    }),
                    // second b (for error estimation)
                    Nd4j.create(new double[]{
                            71.0 / 57600.0, 0.0, -71.0 / 16695.0, 71.0 / 1920.0, -17253.0 / 339200.0, 22.0 / 525.0, -1.0 / 40.0
                    }),
                    // c
                    Nd4j.create(new double[]{
                            1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0
                    }));

    private final AdaptiveRungeKuttaSolver solver;

    private final static class MseComputation implements AdaptiveRungeKuttaSolver.MseComputation {

        private final SolverConfig config;

        private MseComputation(SolverConfig config) {
            this.config = config;
        }

        @Override
        public INDArray estimateMse(
                final INDArray yDotK,
                final INDArray y0,
                final INDArray y1,
                final INDArray h
        ) {
            // TODO: Test remove zero row from bStar and yDotK and see if there are net gains
            final INDArray errSum = butcherTableu.bStar.mmul(yDotK);
            final INDArray yScale = max(abs(y0), abs(y1));
            final INDArray tol = yScale.muli(config.getRelTol()).addi(config.getAbsTol());
            final INDArray ratio = errSum.divi(tol);
            final INDArray error = ratio.muli(ratio);
            return sqrt(error.mean()).muli(h).detach();
        }
    }

    public DormandPrince54Solver(SolverConfig config) {
        solver = new AdaptiveRungeKuttaSolver(config, butcherTableu, 5, new MseComputation(config));
    }

    @Override
    public INDArray integrate(FirstOrderEquation equation, INDArray t, INDArray y0, INDArray yOut) {
        return solver.integrate(equation, t, y0, yOut);
    }


    public static void main(String[] args) {
        INDArray a = Nd4j.randn(new long[] {1});
        INDArray b = Nd4j.randn(new long[] {1});

        INDArray yt = Nd4j.randn(new long[] {10});


        System.out.println(method1(a,b,yt));
        System.out.println(method2(a,b,yt));
    }

   private static INDArray method1(INDArray a, INDArray b, INDArray yt) {
        final INDArray scal = abs(yt).mul(b).add(a);
        final INDArray ratio = yt.div(scal);
        return ratio.muli(ratio).sum();
   }

   private static INDArray method2(INDArray a, INDArray b, INDArray yt) {
        final INDArray scal = abs(yt).mul(b).add(a);
        final INDArray scal2 = scal.mul(scal);
        final INDArray yt2 = yt.mul(yt);
        return yt2.div(scal2).sum();
   }

    private static double estimateErrorReference(
            final double[][] yDotK,
            final double[] y0,
            final double[] y1,
            final double h,
            final double[] E,
            double scalAbsoluteTolerance,
            double scalRelativeTolerance
    ) {

        double error = 0;

        for (int j = 0; j < yDotK[0].length; ++j) {
            final double errSum = E[0] * yDotK[0][j] + E[2] * yDotK[2][j] +
                    E[3] * yDotK[3][j] + E[4] * yDotK[4][j] +
                    E[5] * yDotK[5][j] + E[6] * yDotK[6][j];

            final double yScale = FastMath.max(FastMath.abs(y0[j]), FastMath.abs(y1[j]));
            final double tol =
                    (scalAbsoluteTolerance + scalRelativeTolerance * yScale);
            final double ratio = h * errSum / tol;
            error += ratio * ratio;

        }
        return FastMath.sqrt(error / yDotK[0].length);
    }
}
