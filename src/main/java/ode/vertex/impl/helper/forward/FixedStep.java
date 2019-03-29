package ode.vertex.impl.helper.forward;

import ode.solve.api.FirstOrderMultiStepSolver;
import ode.solve.api.FirstOrderSolver;
import ode.solve.impl.InterpolatingMultiStepSolver;
import ode.solve.impl.SingleSteppingMultiStepSolver;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * {@link OdeHelperForward} with a fixed given sequence of time steps to evaluate the ODE for.
 *
 * @author Christian Skarby
 */
public class FixedStep implements OdeHelperForward {

    private final OdeHelperForward helper;

    public FixedStep(FirstOrderSolver solver, INDArray time, boolean interpolateIfMultiStep) {
        if(time.length() > 2) {
            final FirstOrderMultiStepSolver multiStepSolver = interpolateIfMultiStep ?
                    new InterpolatingMultiStepSolver(solver) :
                    new SingleSteppingMultiStepSolver(solver);
            helper = new MultiStep(multiStepSolver, time);
        } else {
            helper = new SingleStep(solver, time);
        }
    }

    @Override
    public INDArray solve(ComputationGraph graph, LayerWorkspaceMgr wsMgr, GraphInput input) {
        return helper.solve(graph, wsMgr, input);
    }
}
