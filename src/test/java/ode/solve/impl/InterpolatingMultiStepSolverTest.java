package ode.solve.impl;

import ode.solve.CircleODE;
import ode.solve.api.FirstOrderEquation;
import ode.solve.api.FirstOrderSolver;
import ode.solve.conf.SolverConfig;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import static org.junit.Assert.assertArrayEquals;

/**
 * Test cases for {@link InterpolatingMultiStepSolver}
 *
 * @author Christian Skarby
 */
public class InterpolatingMultiStepSolverTest {

    /**
     * Test that output from a {@link SingleSteppingMultiStepSolver} is the same as the output from an
     * {@link InterpolatingMultiStepSolver} when solving the {@link CircleODE} when using the same input.
     */
    @Test
    public void integrateCircle() {
        final Pair<INDArray, INDArray> multiAndInterp = solveCircleMultiInterpol();

        final INDArray multi = multiAndInterp.getFirst();
        final INDArray interp = multiAndInterp.getSecond();

        for(int i = 0; i < multi.columns(); i++)
            assertArrayEquals("Solutions are different in column " + i +"!!",
                    multi.getColumn(i).toDoubleVector(),
                    interp.getColumn(i).toDoubleVector(),
                    1e-4);
    }

    private static Pair<INDArray, INDArray> solveCircleMultiInterpol() {
        final FirstOrderSolver singleStepSolver = new DormandPrince54Solver(
                new SolverConfig(1e-7, 1e-7, 1e-10, 1e2));

        final double omega = 5.67;
        final FirstOrderEquation equation = new CircleODE(new double[] {1.23, 4.56}, omega);

        final int nrofSteps = 25;
        final INDArray y0 = Nd4j.create(new double[] {-5.6, 7.3});
        final INDArray ySingle = y0.dup();
        final INDArray yMulti = Nd4j.repeat(ySingle,nrofSteps-1).reshape(nrofSteps-1, y0.length()).assign(0);
        final INDArray t = Nd4j.linspace(-Math.PI/omega,Math.PI/omega , nrofSteps);

        final INDArray expected = new SingleSteppingMultiStepSolver(singleStepSolver).integrate(equation, t, y0, yMulti.dup()).transposei();
        final INDArray actual = new InterpolatingMultiStepSolver(singleStepSolver).integrate(equation, t, y0, yMulti.dup()).transposei();

        return new Pair<>(expected, actual);
    }
}