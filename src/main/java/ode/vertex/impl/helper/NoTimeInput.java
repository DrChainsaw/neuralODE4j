package ode.vertex.impl.helper;

import ode.vertex.impl.helper.backward.AugmentedDynamics;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.primitives.Pair;

import java.util.ArrayList;
import java.util.List;

/**
 * {@link GraphInputOutput} which does not use time as input to {@link org.deeplearning4j.nn.graph.ComputationGraph}.
 * Note that this is independent of where the time to evaluate the ODE comes from.
 *
 * @author Christian Skarby
 */
public class NoTimeInput implements GraphInputOutput {

    private final INDArray[] inputs;

    public NoTimeInput(INDArray[] inputs) {
        this.inputs = inputs;
    }

    @Override
    public void update(List<INDArray> gradients, AugmentedDynamics augmentedDynamics) {
        augmentedDynamics.updateZAdjoint(gradients);
        augmentedDynamics.tAdjoint().assign(0);
    }

    @Override
    public INDArray[] getInputsFrom(INDArray y, INDArray t) {
        int lastInd = 0;
        for (int i = 0; i < inputs.length; i++) {
            INDArray input = inputs[i];
            final INDArray tmp = y.get(NDArrayIndex.interval(lastInd, lastInd + input.length()));
            lastInd += input.length();
            inputs[i] = tmp.reshape(input.shape());
        }
        return inputs;
    }

    @Override
    public INDArray y0() {
        return inputs[0];
    }

    @Override
    public Pair<GraphInputOutput, INDArray> removeInput(int index) {
        final List<INDArray> inputsToKeep = new ArrayList<>();
        for (int i = 0; i < inputs.length; i++) {
            if (i != index) {
                inputsToKeep.add(inputs[i]);
            }
        }
        return new Pair<>(new NoTimeInput(inputsToKeep.toArray(new INDArray[0])), inputs[index]);
    }
}
