package ode.vertex.conf.helper.forward;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.layers.Convolution3D;

import java.util.ArrayList;
import java.util.List;

/**
 * Adds a time dimension to output types from a given {@link OutputTypeHelper} given an input time type
 *
 * @author Christian Skarby
 */
public class OutputTypeAddTimeAsDimension implements OutputTypeHelper {

    private final int timeInputIndex;
    private final OutputTypeHelper sourceHelper;

    public OutputTypeAddTimeAsDimension(int timeInputIndex, OutputTypeHelper sourceHelper) {
        this.timeInputIndex = timeInputIndex;
        this.sourceHelper = sourceHelper;
    }

    @Override
    public InputType getOutputType(ComputationGraphConfiguration conf, InputType... vertexInputs) throws InvalidInputTypeException {
        List<InputType> inputTypeList = new ArrayList<>();
        InputType time = vertexInputs[timeInputIndex];
        for (int i = 0; i < vertexInputs.length; i++) {
            if (i != timeInputIndex) {
                inputTypeList.add(vertexInputs[i]);
            }
        }

        InputType outputs = sourceHelper.getOutputType(conf, inputTypeList.toArray(new InputType[0]));

        if(time.getType() != InputType.Type.FF) {
            throw new IllegalArgumentException("Time must be 1D!");
        }

        return addTimeDim(outputs, time);
    }

    private InputType addTimeDim(InputType type, InputType timeDim) {
        switch (type.getType()) {
            case FF: return InputType.recurrent(type.arrayElementsPerExample(), timeDim.arrayElementsPerExample());
            case CNN:
                InputType.InputTypeConvolutional convType = (InputType.InputTypeConvolutional)type;
                return InputType.convolutional3D(Convolution3D.DataFormat.NDHWC,
                        timeDim.arrayElementsPerExample(),
                        convType.getHeight(),
                        convType.getWidth(),
                        convType.getChannels());
            default: throw new InvalidInputTypeException("Input type not supported with time as input!");
        }
    }
}
