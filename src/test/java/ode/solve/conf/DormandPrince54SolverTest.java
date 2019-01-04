package ode.solve.conf;

import ode.solve.CircleODE;
import ode.solve.api.FirstOrderSolver;
import ode.solve.api.FirstOrderSolverConf;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link DormandPrince54Solver}
 *
 * @author Christian Skarby
 */
public class DormandPrince54SolverTest extends AbstractConfTest {

    @Override
    protected FirstOrderSolverConf createConf() {
        return new DormandPrince54Solver();
    }

    /**
     * Test that instances seem to work
     */
    @Test
    public void instantiate() {
        final CircleODE equation = new CircleODE(new double[]{1.23, 4.56}, 0.666);

        final DormandPrince54Solver conf =
                new DormandPrince54Solver();

        final FirstOrderSolver reference = new ode.solve.impl.DormandPrince54Solver(conf.getConfig());
        final FirstOrderSolver test = conf.instantiate();

        final INDArray y0 = Nd4j.create(new double[]{3, -5});
        final INDArray y = Nd4j.create(1, 2);
        final INDArray t = Nd4j.create(new double[] {-0.2, 0.4});

        assertEquals("Incorrect solution!", reference.integrate(equation, t, y0, y.dup()), test.integrate(equation, t, y0, y.dup()));
    }
}