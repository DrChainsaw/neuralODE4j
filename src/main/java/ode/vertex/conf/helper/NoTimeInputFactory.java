package ode.vertex.conf.helper;

import lombok.EqualsAndHashCode;
import ode.vertex.impl.helper.GraphInputOutput;
import ode.vertex.impl.helper.NoTimeInput;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Creates {@link NoTimeInput}
 *
 * @author Christian Skarby
 */
@EqualsAndHashCode
public class NoTimeInputFactory implements GraphInputOutputFactory {

    @Override
    public GraphInputOutput create(INDArray[] input) {
        return new NoTimeInput(input);
    }

    @Override
    public GraphInputOutputFactory clone() {
        return new NoTimeInputFactory();
    }
}
