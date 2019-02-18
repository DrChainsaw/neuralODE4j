package examples.spiral.vertex.conf;


import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Reverses the operation of a {@link TimeAsBatch} by making 2D input 3D.
 *
 * @author Christian Skarby
 */
public class ReverseTimeAsBatch extends GraphVertex {

    private final long nrofTimeSteps;

    public ReverseTimeAsBatch(long nrofTimeSteps) {
        this.nrofTimeSteps = nrofTimeSteps;
    }

    @Override
    public GraphVertex clone() {
        return new ReverseTimeAsBatch(nrofTimeSteps);
    }

    @Override
    public boolean equals(Object o) {
        if(!(o instanceof ReverseTimeAsBatch)) {
            return false;
        }

        return ((ReverseTimeAsBatch)o).nrofTimeSteps == nrofTimeSteps;
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
        return new examples.spiral.vertex.impl.ReverseTimeAsBatch(graph, name, idx, nrofTimeSteps);
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType... vertexInputs) throws InvalidInputTypeException {
        if(vertexInputs.length != 1) {
            throw new IllegalArgumentException("Only one input supported!");
        }
        if(vertexInputs[0].getType() != InputType.Type.FF) {
            throw new IllegalArgumentException("Input type must be FF!");
        }

        return InputType.recurrent(vertexInputs[0].arrayElementsPerExample() / nrofTimeSteps, nrofTimeSteps);
    }

    @Override
    public MemoryReport getMemoryReport(InputType... inputTypes) {
        //No working memory in addition to output activations
        return new LayerMemoryReport.Builder(null, TimeAsBatch.class, inputTypes[0], getOutputType(-1, inputTypes))
                .standardMemory(0, 0).workingMemory(0, 0, 0, 0).cacheMemory(0, 0).build();

    }
}
