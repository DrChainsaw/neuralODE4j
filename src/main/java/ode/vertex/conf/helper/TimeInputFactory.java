package ode.vertex.conf.helper;

import lombok.EqualsAndHashCode;
import ode.vertex.impl.helper.GraphInputOutput;
import ode.vertex.impl.helper.NoTimeInput;
import ode.vertex.impl.helper.TimeInput;
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
}
