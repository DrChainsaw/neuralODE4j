package ode.vertex.impl.helper.backward;

import ode.solve.api.FirstOrderSolver;
import ode.vertex.impl.helper.GraphInputOutput;
import ode.vertex.impl.helper.backward.timegrad.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

/**
 * {@link OdeHelperBackward} which uses one of the input {@link INDArray}s as the time steps to evaluate the ODE for
 *
 * @author Christian Skarby
 */
public class InputStepAdjoint implements OdeHelperBackward {

    private final FirstOrderSolver solver;
    private final int timeIndex;
    private final boolean needTimeGradient;

    public InputStepAdjoint(FirstOrderSolver solver, int timeIndex, boolean needTimeGradient) {
        this.solver = solver;
        this.timeIndex = timeIndex;
        this.needTimeGradient = needTimeGradient;
    }


    @Override
    public INDArray[] solve(ComputationGraph graph, InputArrays input, MiscPar miscPars) {
        final Pair<GraphInputOutput, INDArray> result = input.getGraphInputOutput().removeInput(timeIndex);

        final InputArrays newInput = new InputArrays(
                result.getFirst(),
                input.getLastOutput(),
                input.getLossGradient(),
                input.getRealGradientView()
        );

        final INDArray time = result.getSecond();
        if (time.length() > 2) {
            final MultiStepTimeGrad.Factory factory = needTimeGradient ?
                    new CalcMultiStepTimeGrad.Factory(time, timeIndex) :
                    new ZeroMultiStepTimeGrad.Factory(time, timeIndex);

            return new MultiStepAdjoint(solver, time, factory).solve(graph, newInput, miscPars);
        }

        final TimeGrad.Factory factory = needTimeGradient ?
                new CalcTimeGrad.Factory(input.getLossGradient(), timeIndex) :
                new ZeroTimeGrad.Factory(timeIndex);

        return new SingleStepAdjoint(solver, time, factory).solve(graph, newInput, miscPars);
    }
}
