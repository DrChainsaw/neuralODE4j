package ode.vertex.impl.helper.backward;

import ode.solve.api.FirstOrderSolver;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.List;

/**
 * {@link OdeHelperBackward} which uses one of the input {@link INDArray}s as the time steps to evaluate the ODE for
 *
 * @author Christian Skarby
 */
public class InputStepAdjoint implements OdeHelperBackward {

    private final FirstOrderSolver solver;
    private final int timeIndex;

    public InputStepAdjoint(FirstOrderSolver solver, int timeIndex) {
        this.solver = solver;
        this.timeIndex = timeIndex;
    }

    @Override
    public INDArray[] solve(ComputationGraph graph, InputArrays input, MiscPar miscPars) {
        final INDArray time = input.getLastInputs()[timeIndex];
        final List<INDArray> notTimeInputs = new ArrayList<>();
        for(int i = 0; i < input.getLastInputs().length; i++) {
            if(i != timeIndex) {
                notTimeInputs.add(input.getLastInputs()[i]);
            }
        }
        final InputArrays newInput = new InputArrays(
                notTimeInputs.toArray(new INDArray[0]),
                input.getLastOutput(),
                input.getLossGradient(),
                input.getLossGradientTime(),
                input.getRealGradientView()
        );

        if(time.length() > 2) {
            return new MultiStepAdjoint(solver, time, timeIndex).solve(graph, newInput, miscPars);
        }

        return new SingleStepAdjoint(solver, time, timeIndex).solve(graph, newInput, miscPars);
    }
}
