package ode.solve.conf;

import ode.solve.api.FirstOrderSolverConf;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Test cases for {@link DummyIteration}
 *
 * @author Christian Skarby
 */
public class DummyIterationTest extends AbstractConfTest{

    @Override
    protected FirstOrderSolverConf createConf() {
        return new DummyIteration(17);
    }

    /**
     * Test that instances work
     */
    @Test
    public void instantiate() {
        final int expected = 5;
        final ProbeEquation equation = new ProbeEquation();
        new DummyIteration(expected).instantiate().integrate(equation,
                Nd4j.create(new double[] {0, 1}),
                Nd4j.create(3),
                Nd4j.create(3));
        equation.assertNrofCalls(expected);
    }

}