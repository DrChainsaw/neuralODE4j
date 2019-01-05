package ode.solve.conf;

import ode.solve.api.FirstOrderEquation;
import ode.solve.api.FirstOrderSolver;
import ode.solve.api.FirstOrderSolverConf;
import ode.solve.api.StepListener;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/**
 * Test cases for {@link NanWatchSolver}
 *
 * @author Christian Skarby
 */
public class NanWatchSolverTest extends AbstractConfTest {


    @Override
    protected FirstOrderSolverConf createConf() {
        return new NanWatchSolver(new DummyIteration(1));
    }

    /**
     * Test that the instance does not interrupt when no NaNs are present
     */
    @Test
    public void instantiateNoNan() {
        final INDArray y0 = Nd4j.linspace(0, 9, 10);
        final MockSolverConf mockConf = new MockSolverConf();
        final FirstOrderSolver solver = new NanWatchSolver(mockConf).instantiate();

        assertEquals("Incorrect output!", y0, solver.integrate(new MockEquation(),
                Nd4j.scalar(0), y0, Nd4j.zeros(y0.shape())));
    }

    /**
     * Test that the instance works
     */
    @Test(expected = IllegalStateException.class)
    public void instantiateNanIn() {
        final INDArray y0 = Nd4j.linspace(0, 9, 10);
        y0.putScalar(3, Double.NaN);
        final MockSolverConf mockConf = new MockSolverConf();
        final FirstOrderSolver solver = new NanWatchSolver(mockConf).instantiate();

        solver.integrate(new MockEquation(),
                Nd4j.scalar(0), y0, Nd4j.zeros(y0.shape())); // Throws exception!
    }


    /**
     * Mock {@link FirstOrderSolverConf} for testing
     *
     * @author Christian Skarby
     */
    static class MockSolverConf implements FirstOrderSolverConf {

        @Override
        public FirstOrderSolver instantiate() {
            return new FirstOrderSolver() {
                @Override
                public INDArray integrate(FirstOrderEquation equation, INDArray t, INDArray y0, INDArray yOut) {
                    return equation.calculateDerivative(
                            y0,
                            t,
                            yOut);
                }

                @Override
                public void addListener(StepListener... listeners) {
                    fail("Not used!");
                }

                @Override
                public void clearListeners(StepListener... listeners) {
                    fail("Not used!");
                }
            };
        }

        @Override
        public FirstOrderSolverConf clone() {
            return new MockSolverConf();
        }

        @Override
        public boolean equals(Object obj) {
            return obj instanceof MockSolverConf;
        }

        @Override
        public void addListeners(StepListener... listeners) {
            fail("Not used!");
        }

        @Override
        public void clearListeners(StepListener... listeners) {
            fail("Not used!");
        }
    }

    /**
     * Mock {@link FirstOrderEquation} for testing purposes
     *
     * @author Christian Skarby
     */
    private static class MockEquation implements FirstOrderEquation {
        @Override
        public INDArray calculateDerivative(INDArray y, INDArray t, INDArray fy) {
            return fy.assign(y);
        }
    }
}