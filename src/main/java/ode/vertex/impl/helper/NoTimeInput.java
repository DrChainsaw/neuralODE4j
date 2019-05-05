package ode.vertex.impl.helper;

import ode.vertex.impl.helper.backward.AugmentedDynamics;
import org.nd4j.linalg.api.ndarray.INDArray;
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

    private INDArray[] inputs;

    public NoTimeInput(INDArray[] inputs) {
        this.inputs = inputs;
    }

    @Override
    public void update(List<INDArray> gradients, AugmentedDynamics augmentedDynamics) {
        augmentedDynamics.updateZAdjoint(gradients);
    }

    @Override
    public INDArray[] getInputsFrom(INDArray y, INDArray t) {
        if(inputs.length != 1) {
            throw new IllegalArgumentException("Only single input supported!");
        }
        inputs[0] = y.reshape(inputs[0].shape());
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
