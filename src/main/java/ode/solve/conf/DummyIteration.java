package ode.solve.conf;

import lombok.Data;
import lombok.EqualsAndHashCode;
import ode.solve.api.FirstOrderSolver;
import ode.solve.api.FirstOrderSolverConf;
import ode.solve.api.StepListener;
import ode.solve.impl.util.AggStepListener;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Configuration of a {@link ode.solve.impl.DummyIteration}.
 *
 * @author Christian Skarby
 */
@Data
public class DummyIteration implements FirstOrderSolverConf {

    private final int nrofIterations;

    @EqualsAndHashCode.Exclude
    private final AggStepListener listener = new AggStepListener();

    public DummyIteration(@JsonProperty("nrofIterations") int nrofIterations) {
        this.nrofIterations = nrofIterations;
    }

    @Override
    public FirstOrderSolver instantiate() {
        ode.solve.impl.DummyIteration dummyIter = new ode.solve.impl.DummyIteration(() -> nrofIterations);
        dummyIter.addListener(listener);
        return dummyIter;
    }

    @Override
    public FirstOrderSolverConf clone() {
        return new DummyIteration(nrofIterations);
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
