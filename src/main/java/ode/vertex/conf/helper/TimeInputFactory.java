package ode.vertex.conf.helper;

import lombok.EqualsAndHashCode;
import ode.vertex.impl.helper.GraphInputOutput;
import ode.vertex.impl.helper.NoTimeInput;
import ode.vertex.impl.helper.TimeInput;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Creates {@link TimeInput}
 *
 * @author Christian Skarby
 */
@EqualsAndHashCode
public class TimeInputFactory implements GraphInputOutputFactory {

    @Override
    public GraphInputOutput create(INDArray[] input) {
        return new TimeInput(new NoTimeInput(input));
    }

    @Override
    public GraphInputOutputFactory clone() {
        return new TimeInputFactory();
    }

    @Override
    public InputType[] getInputType(InputType... vertexInputs) {
        final InputType[] graphInputs = new InputType[vertexInputs.length +1];
        System.arraycopy(vertexInputs, 0, graphInputs, 0, vertexInputs.length);
        graphInputs[vertexInputs.length] = InputType.feedForward(1);
        return graphInputs;
    }
}
