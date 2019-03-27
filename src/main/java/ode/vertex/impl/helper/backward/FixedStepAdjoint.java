package ode.vertex.impl.helper.backward;

import ode.solve.api.FirstOrderSolver;
import ode.vertex.impl.helper.backward.timegrad.NoMultiStepTimeGrad;
import ode.vertex.impl.helper.backward.timegrad.NoTimeGrad;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * {@link OdeHelperBackward} with a fixed given sequence of time steps to evaluate the ODE for.
 *
 * @author Christian Skarby
 */
public class FixedStepAdjoint implements OdeHelperBackward {

    private final OdeHelperBackward helper;

    public FixedStepAdjoint(FirstOrderSolver solver, INDArray time) {
        if(time.length() > 2) {
            helper = new MultiStepAdjoint(solver, time, NoMultiStepTimeGrad.factory);
        } else {
            helper = new SingleStepAdjoint(solver, time, NoTimeGrad.factory);
        }
    }

    @Override
    public INDArray[] solve(ComputationGraph graph, InputArrays input, MiscPar miscPars) {
        return helper.solve(graph, input, miscPars);
    }
}
