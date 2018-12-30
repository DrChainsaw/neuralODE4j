package ode.solve.impl;

import ode.solve.CircleODE;
import ode.solve.api.FirstOrderSolver;
import ode.solve.commons.FirstOrderSolverAdapter;
import ode.solve.impl.listen.StepListener;
import ode.solve.impl.util.SolverConfig;
import org.apache.commons.math3.ode.nonstiff.DormandPrince54Integrator;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
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

    private static DataBuffer.Type prevType;

    /**
     * Set datatype to double to get more similar results since reference implementation uses double
     */
    @BeforeClass
    public static void setDataType() {
        prevType = Nd4j.dataType();
        Nd4j.setDataType(DataBuffer.Type.DOUBLE);
    }

    /**
     * Reset data type back to previous value
     */
    @AfterClass
    public static void resetDataType() {
        Nd4j.setDataType(prevType);
    }

    /**
     * Test that result is the same as the reference implementation for the {@link CircleODE} problem
     */
    @Test
    public void solveCircleForward() {
        final INDArray ts = Nd4j.create(new double[]{-2, 5});
        solveCircle(ts);
    }

    /**
     * Test that result is the same as the reference implementation for the {@link CircleODE} problem
     */
    @Test
    public void solveCircleBackward() {
        final INDArray ts = Nd4j.create(new double[]{3, -6});
        solveCircle(ts);
    }

    private void solveCircle(INDArray ts) {
        final CircleODE equation = new CircleODE(new double[]{1.23, 4.56}, 0.666);

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

        assertEquals("Incorrect solution!", reference.integrate(equation, ts, y0, y), test.integrate(equation, ts, y0, y));

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
        public void step(INDArray currTime, INDArray step, INDArray error, INDArray y) {
            times.add(currTime.detach());
        }

        @Override
        public void done() {

        }
    }
}