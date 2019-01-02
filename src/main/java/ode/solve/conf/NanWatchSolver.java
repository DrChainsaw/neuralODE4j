package ode.solve.conf;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import ode.solve.api.FirstOrderSolver;
import ode.solve.api.FirstOrderSolverConf;

/**
 * Configuration for a NanWatchSolver.
 *
 * @author Christian Skarby
 */
@Data
@Builder
@AllArgsConstructor
public class NanWatchSolver implements FirstOrderSolverConf {

    private final FirstOrderSolverConf confToWatch;

    @Override
    public FirstOrderSolver instantiate() {
        return new ode.solve.impl.NanWatchSolver(confToWatch.instantiate());
    }

    @Override
    public NanWatchSolver clone() {
        return new NanWatchSolver(confToWatch.clone());
    }
}
