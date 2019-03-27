package ode.vertex.impl.helper.forward;

import ode.solve.api.FirstOrderSolver;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.List;

/**
 * {@link OdeHelperForward} which uses one of the input {@link INDArray}s as the time steps to evaluate the ODE for
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
    public INDArray solve(ComputationGraph graph, LayerWorkspaceMgr wsMgr, INDArray[] inputs) {
        final List<INDArray> notTimeInputs = new ArrayList<>();
        for (int i = 0; i < inputs.length; i++) {
            if (i != timeInputIndex) {
                notTimeInputs.add(inputs[i]);
            }
        }
        return new FixedStep(
                solver,
                inputs[timeInputIndex],
                interpolateIfMultiStep)
                .solve(graph, wsMgr, notTimeInputs.toArray(new INDArray[0]));
    }
}
