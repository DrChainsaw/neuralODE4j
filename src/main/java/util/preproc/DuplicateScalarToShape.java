package util.preproc;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.Arrays;

/**
 * Duplicates scalar input to given shape. Typically used for when time is used as input to a layer in an OdeVertex.
 *
 * @author Christian Skarby
 */
@Data
@EqualsAndHashCode
public class DuplicateScalarToShape implements InputPreProcessor {

    private final long[] shape;

    public DuplicateScalarToShape() {
        this(new long[] {-1, 1});
    }

    /**
     * Constructor.
     * @param shape Desired shape. Set element 0 to -1 in order to use given mini batch size.
     */
    public DuplicateScalarToShape(@JsonProperty("shape") long[] shape) {
        this.shape = shape;
    }

    @Override
    public INDArray preProcess(INDArray input, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
        if(!input.isScalar()) {
            throw new IllegalArgumentException("Can only process scalar input. Got: " + Arrays.toString(input.shape()));
        }
        long[] tmpShape = getShapeFor(miniBatchSize);
        return workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, tmpShape).assign(input.sumNumber());
    }

    long[] getShapeFor(int miniBatchSize) {
        long[] tmpShape = shape.clone();
        if (tmpShape[0] == -1) {
            tmpShape[0] = miniBatchSize;
        }
        return tmpShape;
    }

    @Override
    public INDArray backprop(INDArray output, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
        return workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, output.sum());
    }

    @Override
    public InputPreProcessor clone() {
        return new DuplicateScalarToShape(shape);
    }

    @Override
    public InputType getOutputType(InputType inputType) {
        long[] tmpShape = getShapeFor(1);
        return InputType.inferInputType(Nd4j.createUninitialized(tmpShape));
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState, int minibatchSize) {
        // Mask a scalar??
        throw new UnsupportedOperationException("Not implemented!");
    }
}
