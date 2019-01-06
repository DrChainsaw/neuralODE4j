package util.preproc;

import org.deeplearning4j.nn.conf.inputs.InputType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;

/**
 * Reshapes input to given shape
 *
 * @author Christian Skarby
 */
public class Reshape implements DataSetPreProcessor {

    private final long[] shapeTemplate;

    public Reshape(InputType inputType) {
        this(inputType.getShape(true));
    }

    public Reshape(long[] shapeTemplate) {
        this.shapeTemplate = shapeTemplate;
    }

    @Override
    public void preProcess(DataSet toPreProcess) {
        final INDArray features = toPreProcess.getFeatures();
        shapeTemplate[0] = features.size(0);
        toPreProcess.setFeatures(features.reshape(shapeTemplate));
    }
}
