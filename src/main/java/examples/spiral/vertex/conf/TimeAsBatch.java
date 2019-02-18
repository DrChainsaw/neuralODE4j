package examples.spiral.vertex.conf;

import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Makes 3D input 2D by making dim 2 (typically time) part of dim 0 (batch dim). Basically a poor mans version of
 * torch.nn.Linear wrt allowing 3D input to Dense layers.
 *
 * @author Christian Skarby
 */
public class TimeAsBatch extends GraphVertex {

    @Override
    public GraphVertex clone() {
        return new TimeAsBatch();
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof TimeAsBatch;
    }

    @Override
    public int hashCode() {
        return 0;
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
        return new examples.spiral.vertex.impl.TimeAsBatch(graph, name, idx);
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType... vertexInputs) throws InvalidInputTypeException {
        if(vertexInputs.length != 1) {
            throw new IllegalArgumentException("Only one input supported!");
        }
        if(vertexInputs[0].getType() != InputType.Type.RNN) {
            throw new IllegalArgumentException("Input type must be RNN!");
        }

        InputType.InputTypeRecurrent inputType = (InputType.InputTypeRecurrent)vertexInputs[0];

        return InputType.feedForward(inputType.getSize());
    }

    @Override
    public MemoryReport getMemoryReport(InputType... inputTypes) {
        //No working memory in addition to output activations
        return new LayerMemoryReport.Builder(null, TimeAsBatch.class, inputTypes[0], getOutputType(-1, inputTypes))
                .standardMemory(0, 0).workingMemory(0, 0, 0, 0).cacheMemory(0, 0).build();

    }
}
