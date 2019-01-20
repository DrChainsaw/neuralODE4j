package ode.vertex.impl.helper.backward;

import ode.solve.api.FirstOrderSolver;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

/**
 * {@link OdeHelperBackward} with a fixed given sequence of time steps to evaluate the ODE for.
 *
 * @author Christian Skarby
 */
public class FixedStepAdjoint implements OdeHelperBackward {

    private final OdeHelperBackward helper;

    public FixedStepAdjoint(FirstOrderSolver solver, INDArray time) {
        if(time.length() > 2) {
            helper = new MultiStepAdjoint(solver, time, -1);
        } else {
            helper = new SingleStepAdjoint(solver, time, -1);
        }
    }

    @Override
    public Pair<Gradient, INDArray[]> solve(ComputationGraph graph, InputArrays input, MiscPar miscPars) {
        return helper.solve(graph, input, miscPars);
    }
}
