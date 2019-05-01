package util.preproc;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Convenience class which concatenates zeros to inputs. Intended use case is to be augment an ODE by the process
 * described in https://arxiv.org/pdf/1904.01681.pdf.
 * <br><br>
 * The same thing can be achieved by many other means, e.g. adding an extra input to the graph.
 *
 * @author Christian Skarby
 */
@Data
@EqualsAndHashCode
public class ConcatZeros implements InputPreProcessor {

    private final long nrofZeros;

    public ConcatZeros(@JsonProperty("nrofZeros") long nrofZeros) {
        this.nrofZeros = nrofZeros;
    }

    @Override
    public INDArray preProcess(INDArray input, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
        final long[] zeroesShape = input.shape().clone();

        zeroesShape[1] = nrofZeros; //Same dimension for all of CNNs, FF, RNNs

        return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, Nd4j.hstack(input, Nd4j.zeros(zeroesShape)));
    }

    @Override
    public INDArray backprop(INDArray output, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
        final INDArrayIndex[] epsView = NDArrayIndex.allFor(output);
        epsView[1] = NDArrayIndex.interval(0, output.size(1) - nrofZeros);

        return workspaceMgr.dup(ArrayType.ACTIVATION_GRAD, output.get(epsView));
    }

    @Override
    public ConcatZeros clone() {
        return new ConcatZeros(nrofZeros);
    }

    @Override
    public InputType getOutputType(InputType inputType) {

        switch (inputType.getType()) {
            case FF:
                return InputType.feedForward(inputType.arrayElementsPerExample() + nrofZeros);
            case RNN:
                InputType.InputTypeRecurrent recIn = (InputType.InputTypeRecurrent) inputType;
                return InputType.recurrent(recIn.getSize() + nrofZeros, recIn.getTimeSeriesLength());
            case CNNFlat:
                InputType.InputTypeConvolutionalFlat convInFlat = (InputType.InputTypeConvolutionalFlat) inputType;
                return InputType.convolutionalFlat(convInFlat.getHeight(), convInFlat.getWidth(), convInFlat.getDepth() + nrofZeros);
            case CNN:
                InputType.InputTypeConvolutional convIn = (InputType.InputTypeConvolutional) inputType;
                return InputType.convolutional(convIn.getHeight(), convIn.getWidth(), convIn.getChannels() + nrofZeros);
            case CNN3D:
                InputType.InputTypeConvolutional3D convIn3D = (InputType.InputTypeConvolutional3D) inputType;
                return InputType.convolutional3D(convIn3D.getDataFormat(), convIn3D.getDepth(), convIn3D.getHeight(), convIn3D.getWidth(), convIn3D.getChannels() + nrofZeros);
            default:
                throw new InvalidInputTypeException(this.getClass() + " can't handle input of type: " + inputType);
        }
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState, int minibatchSize) {
        throw new UnsupportedOperationException("Not supported!");
    }
}
