package ode.vertex.conf;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;

/**
 * Convenience class which concatenates zeros to inputs. Intended use case is to be augment an ODE by the process
 * described in https://arxiv.org/pdf/1904.01681.pdf.
 * <br><br>
 * The same thing can be achieved by many other means, e.g. adding an extra input to the graph.
 *
 * @author Christian Skarby
 */
@Data
@EqualsAndHashCode(callSuper = false)
public class ConcatZerosVertex extends GraphVertex {

    private final long nrofZeros;

    public ConcatZerosVertex(long nrofZeros) {
        this.nrofZeros = nrofZeros;
    }

    @Override
    public GraphVertex clone() {
        return new ConcatZerosVertex(nrofZeros);
    }

    @Override
    public long numParams(boolean backprop) {
        return 0;
    }

    @Override
    public int minVertexInputs() {
        return 1;
    }

    @Override
    public int maxVertexInputs() {
        return 1;
    }

    @Override
    public org.deeplearning4j.nn.graph.vertex.GraphVertex instantiate(ComputationGraph graph, String name, int idx, INDArray paramsView, boolean initializeParams) {
        return new ode.vertex.impl.ConcatZerosVertex(graph, name, idx, nrofZeros);
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType... vertexInputs) throws InvalidInputTypeException {
        if (vertexInputs.length != 1) {
            throw new InvalidInputTypeException(this.getClass() + " only accepts a single input!. Got: " + Arrays.toString(vertexInputs));
        }
        final InputType input = vertexInputs[0];
        switch (input.getType()) {
            case FF:
                return InputType.feedForward(input.arrayElementsPerExample() + nrofZeros);
            case RNN:
                InputType.InputTypeRecurrent recIn = (InputType.InputTypeRecurrent) input;
                return InputType.recurrent(recIn.getSize() + nrofZeros, recIn.getTimeSeriesLength());
            case CNNFlat:
                InputType.InputTypeConvolutionalFlat convInFlat = (InputType.InputTypeConvolutionalFlat) input;
                return InputType.convolutionalFlat(convInFlat.getHeight(), convInFlat.getWidth(), convInFlat.getDepth() + nrofZeros);
            case CNN:
                InputType.InputTypeConvolutional convIn = (InputType.InputTypeConvolutional) input;
                return InputType.convolutional(convIn.getHeight(), convIn.getWidth(), convIn.getChannels() + nrofZeros);
            case CNN3D:
                InputType.InputTypeConvolutional3D convIn3D = (InputType.InputTypeConvolutional3D) input;
                return InputType.convolutional3D(convIn3D.getDataFormat(), convIn3D.getDepth(), convIn3D.getHeight(), convIn3D.getWidth(), convIn3D.getChannels() + nrofZeros);
        }
        throw new InvalidInputTypeException(this.getClass() + " can't handle input of type: " + input);

    }

    @Override
    public MemoryReport getMemoryReport(InputType... inputTypes) {
        InputType outputType = getOutputType(-1, inputTypes);

        //TODO multiple input types
        return new LayerMemoryReport.Builder(null, this.getClass(), inputTypes[0], outputType).standardMemory(0, 0) //No params
                .workingMemory(0, 0, 0, 0) //No working memory in addition to activations/epsilons
                .cacheMemory(0, 0) //No caching
                .build();
    }
}
