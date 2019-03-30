package ode.vertex.conf;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;

/**
 * Duplicates the last input to match the shapes of the other inputs. Main use case is for performing merging or element
 * wise operations with current time from an ODE solver.
 *
 * @author Christian Skarby
 */
@Data
@EqualsAndHashCode(callSuper = false)
public class ShapeMatchVertex extends GraphVertex {

    protected GraphVertex graphVertex;

    public ShapeMatchVertex(GraphVertex graphVertex) {
        this.graphVertex = graphVertex;
        if(graphVertex.maxVertexInputs() < 2) {
            throw new IllegalArgumentException("Must be able to take more than one input! Got: " + graphVertex);
        }
    }

    @Override
    public GraphVertex clone() {
        return new ShapeMatchVertex(graphVertex.clone());
    }

    @Override
    public long numParams(boolean backprop) {
        return graphVertex.numParams(backprop);
    }

    @Override
    public int minVertexInputs() {
        return Math.max(2, graphVertex.minVertexInputs());
    }

    @Override
    public int maxVertexInputs() {
        return graphVertex.maxVertexInputs();
    }

    @Override
    public org.deeplearning4j.nn.graph.vertex.GraphVertex instantiate(ComputationGraph graph, String name, int idx, INDArray paramsView, boolean initializeParams) {
        return new ode.vertex.impl.ShapeMatchVertex(graph, name, idx,
                graphVertex.instantiate(graph, name+"-vertex", idx, paramsView, initializeParams));
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType... vertexInputs) throws InvalidInputTypeException {
        final InputType[] afterDuplicate = getInputTypesAfterDuplication(vertexInputs);
        return graphVertex.getOutputType(layerIndex, afterDuplicate);
    }

    @NotNull
    private InputType[] getInputTypesAfterDuplication(InputType[] vertexInputs) {
        if(vertexInputs.length < 2) {
            throw new IllegalArgumentException("Must have more than one inputs!! Got: " + Arrays.toString(vertexInputs));
        }

        final InputType[] afterDuplicate = vertexInputs.clone();
        afterDuplicate[afterDuplicate.length - 1] = afterDuplicate[0];
        return afterDuplicate;
    }

    @Override
    public MemoryReport getMemoryReport(InputType... inputTypes) {
        return graphVertex.getMemoryReport(getInputTypesAfterDuplication(inputTypes));
    }
}
