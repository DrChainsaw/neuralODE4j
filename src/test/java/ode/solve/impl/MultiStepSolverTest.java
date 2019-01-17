package ode.solve.impl;

import ode.solve.CircleODE;
import ode.solve.api.FirstOrderEquation;
import ode.solve.api.FirstOrderSolver;
import ode.solve.api.StepListener;
import ode.solve.conf.SolverConfig;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import util.plot.Plot;
import util.plot.RealTimePlot;

import static junit.framework.TestCase.assertEquals;

/**
 * Test cases for {@link MultiStepSolver}
 *
 * @author Christian Skarby
 */
public class MultiStepSolverTest {

    /**
     * Test that solving {@link ode.solve.CircleODE} in multiple steps gives the same result as if all steps are done
     * in one solve.
     */
    @Test
    public void integrate() {
        final int nrofSteps = 20;
        final FirstOrderEquation circle = new CircleODE(new double[] {1.23, 4.56}, 1);
        final FirstOrderSolver actualSolver = new DormandPrince54Solver(new SolverConfig(1e-10, 1e-10, 1e-10, 10));

        final INDArray t = Nd4j.linspace(0, Math.PI, nrofSteps);
        final INDArray y0 = Nd4j.create(new double[] {0, 0});
        final INDArray ySingle = y0.dup();
        final INDArray yMulti = Nd4j.repeat(ySingle,nrofSteps-1).reshape(nrofSteps-1, y0.length());

        actualSolver.integrate(circle, t.getColumns(0, nrofSteps-1), y0, ySingle);
        new MultiStepSolver(actualSolver).integrate(circle, t, y0, yMulti);

        assertEquals("Incorrect solution!", ySingle, yMulti.getRow(nrofSteps-2));
    }

    public static void main(String[] args) {
        final int nrofSteps = 20;
        final FirstOrderEquation circle = new CircleODE(new double[] {1.23, 4.56}, 1);
        final FirstOrderSolver actualSolver = new DormandPrince54Solver(new SolverConfig(1e-10, 1e-10, 1e-10, 10));

        final INDArray t = Nd4j.linspace(0, Math.PI, nrofSteps);
        final INDArray y0 = Nd4j.create(new double[] {0, 0});
        final INDArray ySingle = y0.dup();
        final INDArray yMulti = Nd4j.repeat(ySingle,nrofSteps-1).reshape(nrofSteps-1, y0.length());

        final Plot<Double, Double> plot = new RealTimePlot<>("circle", "");
        final StepPlotter stepPlotter = new StepPlotter(plot, "singleStep");
        actualSolver.addListener(stepPlotter);
        actualSolver.integrate(circle, t.getColumns(0, nrofSteps-1), y0, ySingle);
        actualSolver.clearListeners();
        new MultiStepSolver(actualSolver).integrate(circle, t, y0, yMulti);


        plot.createSeries("multiStep");
        plot.plotData("multiStep", y0.getDouble(0), y0.getDouble(1));
        for(int step = 1; step < nrofSteps; step++) {
            plot.plotData("multiStep", yMulti.getDouble(step-1, 0), yMulti.getDouble(step-1, 1));
        }
    }

    private static class StepPlotter implements StepListener {

        private final Plot<Double, Double> plot;
        private final String label;

        private StepPlotter(Plot<Double, Double> plot, String label) {
            this.plot = plot;
            this.label = label;
            plot.createSeries(label);
        }

        @Override
        public void begin(INDArray t, INDArray y0) {

        }

        @Override
        public void step(INDArray currTime, INDArray step, INDArray error, INDArray y) {
            plot.plotData(label, y.getDouble(0), y.getDouble(1));
        }

        @Override
        public void done() {

        }
    }
}