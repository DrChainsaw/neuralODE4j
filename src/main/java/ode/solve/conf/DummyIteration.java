package ode.solve.conf;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import ode.solve.api.FirstOrderSolver;
import ode.solve.api.FirstOrderSolverConf;

/**
 * Configuration of a {@link ode.solve.impl.DummyIteration}.
 *
 * @author Christian Skarby
 */
@Data
@Builder
@AllArgsConstructor
public class DummyIteration implements FirstOrderSolverConf {

    private final int nrofIterations;

    @Override
    public FirstOrderSolver instantiate() {
        return new ode.solve.impl.DummyIteration(() -> nrofIterations);
    }

    @Override
    public FirstOrderSolverConf clone() {
        return new DummyIteration(nrofIterations);
    }
}
