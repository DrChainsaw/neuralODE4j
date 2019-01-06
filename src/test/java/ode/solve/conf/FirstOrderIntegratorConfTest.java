package ode.solve.conf;

import ode.solve.CircleODE;
import ode.solve.api.FirstOrderSolver;
import ode.solve.api.FirstOrderSolverConf;
import ode.solve.commons.FirstOrderSolverAdapter;
import org.apache.commons.math3.ode.nonstiff.DormandPrince54Integrator;
import org.apache.commons.math3.ode.nonstiff.DormandPrince853Integrator;
import org.apache.commons.math3.ode.nonstiff.HighamHall54Integrator;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link FirstOrderIntegratorConf}
 *
 * @author Christian Skarby
 */
public class FirstOrderIntegratorConfTest extends AbstractConfTest {

    @Override
    protected FirstOrderSolverConf createConf() {
        return new FirstOrderIntegratorConf();
    }

    /**
     * Test that the default ({@link DormandPrince54Integrator}) works
     */
    @Test
    public void createDormandPrince54() {
        final FirstOrderSolver solver = new FirstOrderIntegratorConf().instantiate();
        final FirstOrderSolver reference = new FirstOrderSolverAdapter(new DormandPrince54Integrator(1e-10, 100, 1e-3, 1e-3));

        verifySolvers(solver, reference);
    }

    /**
     * Test that creating a {@link DormandPrince853Integrator} from its classname also works
     */
    @Test
    public void createDormandPrince853() {
        final SolverConfig config = new SolverConfig(1e-3, 1e-4, 1e-10, 10);
        final FirstOrderSolver solver = new FirstOrderIntegratorConf(DormandPrince853Integrator.class.getName(), config
                ).instantiate();
        final FirstOrderSolver reference = new FirstOrderSolverAdapter(new DormandPrince853Integrator(
                config.getMinStep(),
                config.getMaxStep(),
                config.getAbsTol(),
                config.getRelTol()));

        verifySolvers(solver, reference);
    }

    /**
     * Test that serialization and deserialization works
     */
    @Test
    public void serializeDeserialize() throws IOException {
        final FirstOrderIntegratorConf conf = new FirstOrderIntegratorConf(HighamHall54Integrator.class.getName(),
                new SolverConfig(1e-5, 1e-7, 1e-10, 10));
        final String json = NeuralNetConfiguration.mapper().writeValueAsString(conf);
        final FirstOrderIntegratorConf newConf = NeuralNetConfiguration.mapper().readValue(json, FirstOrderIntegratorConf.class);

        assertEquals("Incorrect solver!", conf.getIntegratorName(), newConf.getIntegratorName());
        assertEquals("Incorrec absTol!", conf.getConfig().getAbsTol(), newConf.getConfig().getAbsTol(), 1e-10);
        verifySolvers(newConf.instantiate(), conf.instantiate());
    }

    private void verifySolvers(FirstOrderSolver solver, FirstOrderSolver reference) {
        final CircleODE equation = new CircleODE(new double[]{1.23, 4.56}, 0.666);
        final INDArray y0 = Nd4j.create(new double[]{11, 13});
        final INDArray y = Nd4j.create(1, 2);
        final INDArray t = Nd4j.create(new double[]{4.56, 1.23});

        assertEquals("Not same solution!", reference.integrate(equation, t, y0, y), solver.integrate(equation, t, y0, y));
    }

}