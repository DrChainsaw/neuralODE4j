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

    public FixedStep(FirstOrderSolverConf solver, INDArray time) {
        this.solver = solver;
        this.time = time.dup();
    }

    @Override
    public OdeHelperForward forward() {
        return new ode.vertex.conf.helper.forward.FixedStep(solver, time);
    }

    @Override
    public OdeHelperBackward backward() {
        return new FixedStepAdjoint(solver, time);
    }
}
