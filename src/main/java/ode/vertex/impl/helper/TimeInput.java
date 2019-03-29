package ode.vertex.impl.helper;

import ode.vertex.impl.helper.backward.AugmentedDynamics;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

import java.util.List;

/**
 * Adds time as last input index
 *
 * @author Christian Skarby
 */
public class TimeInput implements GraphInputOutput {

    private final GraphInputOutput source;

    public TimeInput(INDArray[] input) {
        this(new NoTimeInput(input));
    }

    public TimeInput(GraphInputOutput source) {
        this.source = source;
    }

    @Override
    public INDArray[] getInputsFrom(INDArray y, INDArray t) {
        final INDArray[] inputs = source.getInputsFrom(y, t);
        final INDArray[] withTime = new INDArray[inputs.length + 1];
        System.arraycopy(inputs, 0, withTime, 0, inputs.length);
        withTime[inputs.length] = t;
        return withTime;
    }

    @Override
    public INDArray y0() {
        return source.y0();
    }

    @Override
    public Pair<GraphInputOutput, INDArray> removeInput(int index) {
        Pair<GraphInputOutput, INDArray> res = source.removeInput(index);
        return new Pair<>(new TimeInput(res.getFirst()), res.getSecond());
    }

    @Override
    public void update(List<INDArray> gradients, AugmentedDynamics augmentedDynamics) {
        augmentedDynamics.updateZAdjoint(gradients.subList(0, gradients.size()-1));
        augmentedDynamics.tAdjoint().assign(gradients.get(gradients.size()-1));
    }
}
