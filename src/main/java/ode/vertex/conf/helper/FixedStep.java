package ode.vertex.conf.helper;

import ode.solve.api.FirstOrderSolverConf;
import ode.vertex.conf.helper.backward.FixedStepAdjoint;
import ode.vertex.conf.helper.backward.OdeHelperBackward;
import ode.vertex.conf.helper.forward.OdeHelperForward;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Configuration for using a {@link ode.solve.api.FirstOrderSolver} inside a ComputationGraph when a fixed predefined
 * set of time steps shall be used when solving the ODE.
 *
 * @author Christian Skarby
 */
public class FixedStep implements OdeHelper {

    private final FirstOrderSolverConf solver;
    private final INDArray time;
    private final boolean interpolateForwardIfMultiStep;

    public FixedStep(FirstOrderSolverConf solver, INDArray time) {
        this(solver, time, false);
    }

    public FixedStep(FirstOrderSolverConf solver, INDArray time, boolean interpolateForwardIfMultiStep) {
        this.solver = solver;
        this.time = time.dup();
        this.interpolateForwardIfMultiStep = interpolateForwardIfMultiStep;
    }

    @Override
    public OdeHelperForward forward() {
        return new ode.vertex.conf.helper.forward.FixedStep(solver, time, interpolateForwardIfMultiStep);
    }

    @Override
    public OdeHelperBackward backward() {
        return new FixedStepAdjoint(solver, time);
    }
}
