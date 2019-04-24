package ode.vertex.impl.helper.forward;

import ode.solve.api.FirstOrderSolver;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

/**
 * {@link OdeHelperForward} which uses one of the input {@link INDArray}s as the time steps to evaluate the ODE for. Note
 * that this is not the same thing as having the ODE function itself depend on time (which is decided by {@link GraphInput}
 * implementation).
 *
 * @author Christian Skarby
 */
public class InputStep implements OdeHelperForward {

    private final FirstOrderSolver solver;
    private final int timeInputIndex;
    private final boolean interpolateIfMultiStep;

    public InputStep(FirstOrderSolver solver, int timeInputIndex, boolean interpolateIfMultiStep) {
        this.solver = solver;
        this.timeInputIndex = timeInputIndex;
        this.interpolateIfMultiStep = interpolateIfMultiStep;
    }

    @Override
    public INDArray solve(ComputationGraph graph, LayerWorkspaceMgr wsMgr, GraphInput input) {
        Pair<? extends GraphInput, INDArray> result = input.removeInput(timeInputIndex);
        return new FixedStep(
                solver,
                result.getSecond(),
                interpolateIfMultiStep)
                .solve(graph, wsMgr, result.getFirst());
    }
}
