package ode.solve.conf;

import lombok.Data;
import ode.solve.api.FirstOrderSolver;
import ode.solve.api.FirstOrderSolverConf;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Configuration of a {@link ode.solve.impl.DummyIteration}.
 *
 * @author Christian Skarby
 */
@Data
public class DummyIteration implements FirstOrderSolverConf {

    private final int nrofIterations;

    public DummyIteration(@JsonProperty("nrofIterations") int nrofIterations) {
        this.nrofIterations = nrofIterations;
    }

    @Override
    public FirstOrderSolver instantiate() {
        return new ode.solve.impl.DummyIteration(() -> nrofIterations);
    }

    @Override
    public FirstOrderSolverConf clone() {
        return new DummyIteration(nrofIterations);
    }
}
