package ode.vertex.impl.helper.backward;

import ode.solve.api.FirstOrderSolver;
import ode.vertex.impl.helper.backward.timegrad.*;
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
    private final boolean needTimeGradient;

    public InputStepAdjoint(FirstOrderSolver solver, int timeIndex, boolean needTimeGradient) {
        this.solver = solver;
        this.timeIndex = timeIndex;
        this.needTimeGradient = needTimeGradient;
    }


    @Override
    public INDArray[] solve(ComputationGraph graph, InputArrays input, MiscPar miscPars) {
        final INDArray time = input.getLastInputs()[timeIndex];
        final List<INDArray> notTimeInputs = new ArrayList<>();
        for (int i = 0; i < input.getLastInputs().length; i++) {
            if (i != timeIndex) {
                notTimeInputs.add(input.getLastInputs()[i]);
            }
        }
        final InputArrays newInput = new InputArrays(
                notTimeInputs.toArray(new INDArray[0]),
                input.getLastOutput(),
                input.getLossGradient(),
                input.getRealGradientView()
        );

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
