package ode.solve.impl.util;

import examples.spiral.listener.PlotDecodedOutput;
import ode.solve.CircleODE;
import ode.solve.api.FirstOrderEquation;
import ode.solve.api.FirstOrderSolver;
import ode.solve.conf.SolverConfig;
import ode.solve.impl.DormandPrince54Solver;
import ode.solve.impl.SingleSteppingMultiStepSolver;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.primitives.Pair;
import util.plot.Plot;
import util.plot.RealTimePlot;

import java.util.Collections;

import static org.junit.Assert.assertArrayEquals;

/**
 * Test cases for {@link InterpolatingStepListener}
 *
 * @author Christian Skarby
 */
public class InterpolatingStepListenerTest {

    /**
     * Verify that the result from using an {@link InterpolatingStepListener} is equivalent to using a
     * {@link SingleSteppingMultiStepSolver} when solving the {@link CircleODE} when using the same input.
     */
    @Test
    public void iterpolateCircle() {
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
        final INDArray tStartEnd = Nd4j.create(new double[] {-Math.PI/omega, Math.PI/omega});
        final INDArray t = Nd4j.linspace(tStartEnd.getDouble(0),tStartEnd.getDouble(1) , nrofSteps);

        final INDArray expected = new SingleSteppingMultiStepSolver(singleStepSolver).integrate(equation, t, y0, yMulti.dup()).transposei();

        singleStepSolver.addListener(new InterpolatingStepListener(t.get(NDArrayIndex.interval(1,nrofSteps)), yMulti));
        final INDArray yLast = y0.dup();
        singleStepSolver.integrate(equation, tStartEnd, y0, yLast);

        return new Pair<>(expected, yMulti.transposei());
    }

    /**
     * Main method for plotting results from both approaches
     * @param args not used
     */
    public static void main(String[] args) {
        final Plot<Double, Double> plot = new RealTimePlot<>("Multi step vs interpol", "");
        final Pair<INDArray, INDArray> multiAndInterp = solveCircleMultiInterpol();

        final INDArray multi = multiAndInterp.getFirst();
        final INDArray interp = multiAndInterp.getSecond();

        // To avoid annoying NPE in swing thread due to xychart not being thread safe
        plot.createSeries("Multi");
        plot.createSeries("Interp");
        plot.createSeries("Multi start");
        plot.createSeries("Interp start");
        plot.createSeries("Multi stop");
        plot.createSeries("Interp stop");

        plot(plot, multi, "Multi");
        plot(plot, interp, "Interp");

        plot.plotData("Multi start", multi.getDouble(0,0), multi.getDouble(1, 0));
        plot.plotData("Interp start", interp.getDouble(0,0), interp.getDouble(1, 0));

        final long lastStep = multi.size(1)-1;
        plot.plotData("Multi stop", multi.getDouble(0,lastStep), multi.getDouble(1, lastStep));
        plot.plotData("Interp stop", interp.getDouble(0,lastStep), interp.getDouble(1, lastStep));

    }

    private static void plot(Plot<Double, Double> plot, INDArray toPlot, String label) {
        new PlotDecodedOutput(plot, label, 0)
                .onForwardPass(null, Collections.singletonMap(label,
                        toPlot.reshape(1, toPlot.rows(), toPlot.columns())));

    }

}