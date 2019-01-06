package ode.solve.conf;

import lombok.Data;
import lombok.EqualsAndHashCode;
import ode.solve.api.FirstOrderSolver;
import ode.solve.api.FirstOrderSolverConf;
import ode.solve.api.StepListener;
import ode.solve.impl.util.AggStepListener;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Configuration for a {@link ode.solve.impl.DormandPrince54Solver}.
 *
 * @author Christian Skarby
 */
@Data
public class DormandPrince54Solver implements FirstOrderSolverConf {

    private final SolverConfig config;

    @EqualsAndHashCode.Exclude
    private final AggStepListener listener = new AggStepListener();

    public DormandPrince54Solver() {
        this(new SolverConfig(1e-3, 1e-3, 1e-10, 100));
    }

    public DormandPrince54Solver(@JsonProperty("config") SolverConfig config) {
        this.config = config;
    }

    @Override
    public FirstOrderSolver instantiate() {
        final ode.solve.impl.DormandPrince54Solver solver =  new ode.solve.impl.DormandPrince54Solver(config);
        solver.addListener(listener);
        return solver;
    }

    @Override
    public DormandPrince54Solver clone()  {
        return new DormandPrince54Solver(new SolverConfig(
                config.getAbsTol(),
                config.getRelTol(),
                config.getMinStep(),
                config.getMaxStep()));
    }

    @Override
    public void addListeners(StepListener... listeners) {
        listener.addListeners(listeners);
    }

    @Override
    public void clearListeners(StepListener... listeners) {
        listener.clearListeners(listeners);
    }
}
