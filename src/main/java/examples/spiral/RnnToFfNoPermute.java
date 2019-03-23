package examples.spiral;

import lombok.val;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.preprocessor.RnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.util.TimeSeriesUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.primitives.Pair;

import java.util.Arrays;

/**
 * Copy of {@link RnnToFeedForwardPreProcessor} without the permute operation
 */
public class RnnToFfNoPermute implements InputPreProcessor {
    private static final int BATCH_DIM = 0;
    private static final int SIZE_DIM = 1;
    private static final int TIME_DIM = 2;

    @Override
    public INDArray preProcess(INDArray input, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
        //Need to reshape RNN activations (3d) activations to 2d (for input into feed forward layer)
        if (input.rank() != 3)
            throw new IllegalArgumentException(
                    "Invalid input: expect NDArray with rank 3 (i.e., activations for RNN layer)");

        if (input.ordering() != 'f' || !Shape.hasDefaultStridesForShape(input))
            input = workspaceMgr.dup(ArrayType.ACTIVATIONS, input, 'f');

        val shape = input.shape();
        INDArray ret;
        if (shape[2] == 1) {
            ret = input.tensorAlongDimension(0, 1, 0); //Edge case: timeSeriesLength=1
        } else {
            ret = input.permute(BATCH_DIM, TIME_DIM, SIZE_DIM).reshape(shape[BATCH_DIM] * shape[TIME_DIM], shape[SIZE_DIM]);
            //ret = input.reshape(input.length()).reshape(shape[BATCH_DIM]*shape[TIME_DIM], shape[SIZE_DIM]);
        }
        return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, ret);
    }

    @Override
    public INDArray backprop(INDArray output, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
        if (output == null)
            return null; //In a few cases: output may be null, and this is valid. Like time series data -> embedding layer
        //Need to reshape FeedForward layer epsilons (2d) to 3d (for use in RNN layer backprop calculations)
        if (output.rank() != 2)
            throw new IllegalArgumentException(
                    "Invalid input: expect NDArray with rank 2 (i.e., epsilons from feed forward layer)");
        if (output.ordering() != 'f' || !Shape.hasDefaultStridesForShape(output))
            output = workspaceMgr.dup(ArrayType.ACTIVATION_GRAD, output, 'f');

        val shape = output.shape();
        INDArray reshaped = output.reshape(miniBatchSize, shape[BATCH_DIM] / miniBatchSize, shape[SIZE_DIM])
                .permutei(BATCH_DIM, TIME_DIM, SIZE_DIM);
        return workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, reshaped);
    }

    @Override
    public RnnToFeedForwardPreProcessor clone() {
        try {
            RnnToFeedForwardPreProcessor clone = (RnnToFeedForwardPreProcessor) super.clone();
            return clone;
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public InputType getOutputType(InputType inputType) {
        if (inputType == null || inputType.getType() != InputType.Type.RNN) {
            throw new IllegalStateException("Invalid input: expected input of type RNN, got " + inputType);
        }

        InputType.InputTypeRecurrent rnn = (InputType.InputTypeRecurrent) inputType;
        return InputType.feedForward(rnn.getSize());
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState,
                                                          int minibatchSize) {
        //Assume mask array is 2d for time series (1 value per time step)
        if (maskArray == null) {
            return new Pair<>(maskArray, currentMaskState);
        } else if (maskArray.rank() == 2) {
            //Need to reshape mask array from [minibatch,timeSeriesLength] to [minibatch*timeSeriesLength, 1]
            return new Pair<>(TimeSeriesUtils.reshapeTimeSeriesMaskToVector(maskArray, LayerWorkspaceMgr.noWorkspaces(), ArrayType.INPUT),  //TODO
                    currentMaskState);
        } else {
            throw new IllegalArgumentException("Received mask array of rank " + maskArray.rank()
                    + "; expected rank 2 mask array. Mask array shape: " + Arrays.toString(maskArray.shape()));
        }
    }
}
