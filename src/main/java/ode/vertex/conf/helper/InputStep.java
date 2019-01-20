package ode.vertex.conf.helper;

import ode.solve.api.FirstOrderSolverConf;
import ode.vertex.conf.helper.backward.InputStepAdjoint;
import ode.vertex.conf.helper.backward.OdeHelperBackward;
import ode.vertex.conf.helper.forward.OdeHelperForward;

/**
 * Configuration for using a {@link ode.solve.api.FirstOrderSolver} inside a {@code ComputationGraph} when time steps for solving
 * the ODE comes as inputs to the {@code GraphVertex} housing the ODE.
 * <br><br>
 *     Example: <br>
 * <pre>
 * graphBuilder.addVertex("odeVertex",
 *     new OdeVertex.Builder("0", new DenseLayer.Builder().nOut(4).build())
 *         .odeConf(new InputStep(solverConf, 1)) // Refers to input "time" on the line below
 *         .build(), "someLayer", "time");
 * </pre>
 *
 * @author Christian Skarby
 */
public class InputStep implements OdeHelper {

    private final FirstOrderSolverConf solverConf;
    private final int timeInputIndex;

    public InputStep(FirstOrderSolverConf solverConf, int timeInputIndex) {
        this.solverConf = solverConf;
        this.timeInputIndex = timeInputIndex;
    }

    @Override
    public OdeHelperForward forward() {
        return new ode.vertex.conf.helper.forward.InputStep(solverConf, timeInputIndex);
    }

    @Override
    public OdeHelperBackward backward() {
        return new InputStepAdjoint(solverConf, timeInputIndex);
    }
}
