package ode.solve.conf;

import lombok.Data;
import ode.solve.api.FirstOrderSolver;
import ode.solve.api.FirstOrderSolverConf;
import ode.solve.api.StepListener;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.Objects;

/**
 * Configuration for a NanWatchSolver.
 *
 * @author Christian Skarby
 */
@Data
public class NanWatchSolver implements FirstOrderSolverConf {

    private final FirstOrderSolverConf confToWatch;

    public NanWatchSolver(@JsonProperty("confToWatch") FirstOrderSolverConf confToWatch) {
        this.confToWatch = Objects.requireNonNull(confToWatch);
    }

    @Override
    public FirstOrderSolver instantiate() {
        return new ode.solve.impl.NanWatchSolver(confToWatch.instantiate());
    }

    @Override
    public NanWatchSolver clone() {
        return new NanWatchSolver(confToWatch.clone());
    }

    @Override
    public boolean equals(Object obj) {
        if(!(obj instanceof NanWatchSolver)) {
            return false;
        }
        return ((NanWatchSolver)obj).confToWatch.equals(confToWatch);
    }

    @Override
    public void addListeners(StepListener... listeners) {
        confToWatch.addListeners(listeners);
    }

    @Override
    public void clearListeners(StepListener... listeners) {
        confToWatch.clearListeners(listeners);
    }
}
