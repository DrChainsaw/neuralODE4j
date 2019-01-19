package ode.vertex.impl.helper.forward;

import ode.solve.api.FirstOrderSolver;
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

    public FixedStep(FirstOrderSolver solver, INDArray time) {
        if(time.length() > 2) {
            helper = new MultiStep(solver, time);
        } else {
            helper = new SingleStep(solver, time);
        }
    }

    @Override
    public INDArray solve(ComputationGraph graph, LayerWorkspaceMgr wsMgr, INDArray[] inputs) {
        return helper.solve(graph, wsMgr, inputs);
    }
}
