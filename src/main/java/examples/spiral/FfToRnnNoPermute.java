package examples.spiral;

import lombok.val;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.util.TimeSeriesUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.primitives.Pair;

import java.util.Arrays;

/**
 * Copy of {@link org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor} without the permute operation
 */
public class FfToRnnNoPermute implements InputPreProcessor {

    private static final int BATCH_DIM = 0;
    private static final int SIZE_DIM = 1;
    private static final int TIME_DIM = 2;

    @Override
    public INDArray preProcess(INDArray input, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
        //Need to reshape FF activations (2d) activations to 3d (for input into RNN layer)
        if (input.rank() != 2)
            throw new IllegalArgumentException(
                    "Invalid input: expect NDArray with rank 2 (i.e., activations for FF layer)");
        if (input.ordering() != 'f' || !Shape.hasDefaultStridesForShape(input))
            input = workspaceMgr.dup(ArrayType.ACTIVATIONS, input, 'f');

        val shape = input.shape();
        final INDArray reshaped = input.reshape(miniBatchSize, shape[BATCH_DIM] / miniBatchSize, shape[SIZE_DIM])
                .permutei(BATCH_DIM, TIME_DIM, SIZE_DIM);
        //final INDArray reshaped = input.reshape(input.length()).reshape(miniBatchSize, shape[SIZE_DIM], shape[BATCH_DIM] / miniBatchSize);
        return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, reshaped);
    }

    @Override
    public INDArray backprop(INDArray output, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
        //Need to reshape RNN epsilons (3d) to 2d (for use in FF layer backprop calculations)
        if (output.rank() != 3)
            throw new IllegalArgumentException(
                    "Invalid input: expect NDArray with rank 3 (i.e., epsilons from RNN layer)");
        if (output.ordering() != 'f' || !Shape.hasDefaultStridesForShape(output))
            output = workspaceMgr.dup(ArrayType.ACTIVATION_GRAD, output, 'f');
        val shape = output.shape();

        INDArray ret;
        if (shape[2] == 1) {
            return output.tensorAlongDimension(0, 1, 0); //Edge case: timeSeriesLength=1
        } else {
            ret = output.permute(BATCH_DIM, TIME_DIM, SIZE_DIM).reshape(shape[BATCH_DIM] * shape[TIME_DIM], shape[SIZE_DIM]);
            //ret = output.reshape(output.length()).reshape(shape[BATCH_DIM] * shape[TIME_DIM], shape[SIZE_DIM]);
        }
        return workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, ret);
    }

    @Override
    public FeedForwardToRnnPreProcessor clone() {
        try {
            FeedForwardToRnnPreProcessor clone = (FeedForwardToRnnPreProcessor) super.clone();
            return clone;
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public InputType getOutputType(InputType inputType) {
        if (inputType == null || (inputType.getType() != InputType.Type.FF
                && inputType.getType() != InputType.Type.CNNFlat)) {
            throw new IllegalStateException("Invalid input: expected input of type FeedForward, got " + inputType);
        }

        if (inputType.getType() == InputType.Type.FF) {
            InputType.InputTypeFeedForward ff = (InputType.InputTypeFeedForward) inputType;
            return InputType.recurrent(ff.getSize());
        } else {
            InputType.InputTypeConvolutionalFlat cf = (InputType.InputTypeConvolutionalFlat) inputType;
            return InputType.recurrent(cf.getFlattenedSize());
        }
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState,
                                                          int minibatchSize) {
        //Assume mask array is 1d - a mask array that has been reshaped from [minibatch,timeSeriesLength] to [minibatch*timeSeriesLength, 1]
        if (maskArray == null) {
            return new Pair<>(maskArray, currentMaskState);
        } else if (maskArray.isVector()) {
            //Need to reshape mask array from [minibatch*timeSeriesLength, 1] to [minibatch,timeSeriesLength]
            return new Pair<>(TimeSeriesUtils.reshapeVectorToTimeSeriesMask(maskArray, minibatchSize),
                    currentMaskState);
        } else {
            throw new IllegalArgumentException("Received mask array with shape " + Arrays.toString(maskArray.shape())
                    + "; expected vector.");
        }
    }
}
