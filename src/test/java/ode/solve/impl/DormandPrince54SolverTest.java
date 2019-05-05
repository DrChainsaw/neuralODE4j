package ode.solve.impl;

import ode.solve.CircleODE;
import ode.solve.api.FirstOrderSolver;
import ode.solve.api.StepListener;
import ode.solve.commons.FirstOrderSolverAdapter;
import ode.solve.conf.SolverConfig;
import ode.solve.impl.util.SolverState;
import org.apache.commons.math3.ode.nonstiff.DormandPrince54Integrator;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link DormandPrince54Solver}
 *
 * @author Christian Skarby
 */
public class DormandPrince54SolverTest {

    private static DataType prevType;
    private static DataType prevFloatType;

    /**
     * Set datatype to double to get more similar results since reference implementation uses double
     */
    @BeforeClass
    public static void setDataType() {
        prevType = Nd4j.dataType();
        prevFloatType = Nd4j.defaultFloatingPointType();
        Nd4j.setDefaultDataTypes(prevType, prevFloatType);
    }

    /**
     * Reset data type back to previous value
     */
    @AfterClass
    public static void resetDataType() {
        Nd4j.setDefaultDataTypes(prevType, prevFloatType);
    }

    /**
     * Test that result is the same as the reference implementation for the {@link CircleODE} problem
     */
    @Test
    public void solveCircleForward() {
        final INDArray ts = Nd4j.create(new double[]{-0.023, 0.0456});
        solveCircle(ts);
    }

    /**
     * Test that result is the same as the reference implementation for the {@link CircleODE} problem
     */
    @Test
    public void solveCircleBackward() {
        final INDArray ts = Nd4j.create(new double[]{0.023, -0.0456});
        solveCircle(ts);
    }

    private void solveCircle(INDArray ts) {
        final CircleODE equation = new CircleODE(new double[]{1.23, 4.56}, 20.666);

        final FirstOrderSolver reference = new FirstOrderSolverAdapter(
                new DormandPrince54Integrator(1e-10, 100, 1e-10, 1e-10));
        final FirstOrderSolver test =
                new DormandPrince54Solver(new SolverConfig(
                        1e-10, 1e-10, 1e-10, 100));

        final StepCounter refCounter = new StepCounter();
        final StepCounter testCounter = new StepCounter();

        reference.addListener(refCounter);
        test.addListener(testCounter);

        final INDArray y0 = Nd4j.create(new double[]{3, -5});
        final INDArray y = Nd4j.create(1, 2);


        assertEquals("Incorrect solution!", reference.integrate(equation, ts, y0, y.dup()), test.integrate(equation, ts, y0, y.dup()));

        assertEquals("Incorrect number of steps", refCounter.times.size(), testCounter.times.size());

        for(int i = 0; i < refCounter.times.size(); i++) {
            final double expected = refCounter.times.get(i).getDouble(0);
            final double actual = testCounter.times.get(i).getDouble(0);
            assertEquals("Incorrect time at step " + i + "!", expected, actual, 1e-8);
        }
    }


    private final class StepCounter implements StepListener {

        private final List<INDArray> times = new ArrayList<>();

        @Override
        public void begin(INDArray t, INDArray y0) {

        }

        @Override
        public void step(SolverState solverState, INDArray step, INDArray error) {
            times.add(solverState.time().detach());
        }

        @Override
        public void done() {

        }
    }
}