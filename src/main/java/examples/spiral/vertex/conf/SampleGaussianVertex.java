package examples.spiral.vertex.conf;

import lombok.Data;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.jackson.annotation.JsonProperty;


/**
 * Takes samples from a Gaussian process where mean and std are inputs, typically from a set of layers acting as a
 * variational auto encoder.
 *
 * @author Christian Skarby
 */
@Data
public class SampleGaussianVertex extends GraphVertex {

    private final long seed;

    public SampleGaussianVertex(@JsonProperty("seed") long seed) {
        this.seed = seed;
    }

    @Override
    public GraphVertex clone() {
        return new SampleGaussianVertex(seed);
    }

    @Override
    public boolean equals(Object o) {
        if(o instanceof SampleGaussianVertex) {
            SampleGaussianVertex other = (SampleGaussianVertex)o;
            return other.seed == this.seed;
        }
        return false;
    }

    @Override
    public int hashCode() {
        return (int)seed;
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
        final Random rng = Nd4j.getRandomFactory().getNewRandomInstance(seed);
        return new examples.spiral.vertex.impl.SampleGaussianVertex(graph, name, idx,  shape -> Nd4j.randn(shape, rng));
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType... vertexInputs) throws InvalidInputTypeException {
        if(vertexInputs.length != 1) {
            throw new IllegalArgumentException(this.getClass().getSimpleName() +" must have one inputs!");
        }

        for(InputType inputType: vertexInputs) {
            if(inputType.getType() != InputType.Type.FF) {
                throw new IllegalArgumentException(this.getClass().getSimpleName() + " only supports feedforward input! Got: " + inputType);
            }

            if(inputType.arrayElementsPerExample() % 2 != 0) {
                throw new IllegalArgumentException(this.getClass().getSimpleName() + " input size must be even! Got: " + inputType);
            }
        }

        return InputType.feedForward(vertexInputs[0].arrayElementsPerExample() / 2);
    }

    @Override
    public MemoryReport getMemoryReport(InputType... inputTypes) {
        InputType outputType = getOutputType(-1, inputTypes);

        return new LayerMemoryReport.Builder(null, SampleGaussianVertex.class, inputTypes[0], outputType).standardMemory(0, 0) //No params
                .workingMemory(0, 0, 0, 0) //No working memory in addition to activations/epsilons
                .cacheMemory(0, 0) //No caching
                .build();
    }
}
