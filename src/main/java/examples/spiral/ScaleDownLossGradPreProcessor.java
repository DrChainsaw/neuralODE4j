package examples.spiral;

import lombok.Data;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Scales the loss gradient with the number of elements in one dimension to compensate for when that dimension is used
 * as batch dimension. For example, when RNN input is fed to a Dense layer, the time steps are permuted into the batch
 * dimension so that the Dense layer now sees a batch size of batchSize*numTimeSteps. However, when
 * {@link org.deeplearning4j.nn.updater.BaseMultiLayerUpdater} scales the gradients, it only scales them with batchSize.
 * <br><br>
 * To counteract this, this preprocessor scales the loss gradient with the size of a given dimension.
 *
 * @author Christian Skarby
 */
@Data
public class ScaleDownLossGradPreProcessor implements InputPreProcessor {

    private final int scaleDim;
    private final InputPreProcessor inputPreProcessor;

    public ScaleDownLossGradPreProcessor(@JsonProperty("scaleDim") int scaleDim,
                                         @JsonProperty("inputPreProcessor") InputPreProcessor inputPreProcessor) {
        this.scaleDim = scaleDim;
        this.inputPreProcessor = inputPreProcessor;
    }

    @Override
    public INDArray preProcess(INDArray input, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
        return inputPreProcessor.preProcess(input, miniBatchSize, workspaceMgr);
    }

    @Override
    public INDArray backprop(INDArray output, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
        return inputPreProcessor.backprop(output.divi(output.size(scaleDim)), miniBatchSize, workspaceMgr);
    }

    @Override
    public InputPreProcessor clone() {
        return new ScaleDownLossGradPreProcessor(scaleDim, inputPreProcessor.clone());
    }

    @Override
    public InputType getOutputType(InputType inputType) {
        return inputPreProcessor.getOutputType(inputType);
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState, int minibatchSize) {
        return inputPreProcessor.feedForwardMaskArray(maskArray, currentMaskState, minibatchSize);
    }
}
