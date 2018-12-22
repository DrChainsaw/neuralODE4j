package ode.conf;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.layers.CnnLossLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Configuration of an ODE block.
 *
 * @author Christian Skarby
 */
public class OdeVertex extends GraphVertex {

    private final ComputationGraphConfiguration conf;
    private final String firstVertex;
    private final String lastVertex;

    public OdeVertex(ComputationGraphConfiguration conf, String firstVertex, String lastVertex) {
        this.conf = conf;
        this.firstVertex = firstVertex;
        this.lastVertex = lastVertex;
    }

    @Override
    public GraphVertex clone() {
        return new OdeVertex(conf.clone(), firstVertex, lastVertex);
    }

    @Override
    public boolean equals(Object o) {
        if(!(o instanceof OdeVertex)) {
            return false;
        }
        return conf.equals(((OdeVertex) o).conf);
    }

    @Override
    public int hashCode() {
        return conf.hashCode();
    }

    @Override
    public long numParams(boolean backprop) {
        return conf.getVertices().values().stream()
                .mapToLong(vertex -> vertex.numParams(backprop))
                .sum();
    }

    @Override
    public int minVertexInputs() {
        return conf.getVertices().get(firstVertex).minVertexInputs();
    }

    @Override
    public int maxVertexInputs() {
        return conf.getVertices().get(firstVertex).maxVertexInputs();
    }

    @Override
    public org.deeplearning4j.nn.graph.vertex.GraphVertex instantiate(ComputationGraph graph, String name, int idx, INDArray paramsView, boolean initializeParams) {
        final ComputationGraph innerGraph = new ComputationGraph(conf);
        innerGraph.init(paramsView, false);
        return new ode.impl.OdeVertex(graph, name, idx, null, null, innerGraph);
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType... vertexInputs) throws InvalidInputTypeException {
        return conf.getLayerActivationTypes(vertexInputs).get(lastVertex);
    }

    @Override
    public MemoryReport getMemoryReport(InputType... inputTypes) {
        return conf.getMemoryReport(inputTypes);
    }

    public static class Builder {

        private final String inputName = this.toString() + "_input";
        private final String outputName = this.toString() + "_output";
        private final ComputationGraphConfiguration.GraphBuilder graphBuilder = new NeuralNetConfiguration.Builder()
                .graphBuilder();

        private String first = null;
        private String last;


        public Builder(String name, Layer layer) {
            graphBuilder
                    .addInputs(inputName)
                    .addLayer(name, layer, inputName);
        }

        /**
         * @see ComputationGraphConfiguration.GraphBuilder#addLayer(String, Layer, String...)
         */
        public Builder addLayer(String name, Layer layer, String... inputs) {
            graphBuilder.addLayer(name, layer, inputs);
            checkFirst(name);
            last = name;
            return this;
        }

        /**
         * @see ComputationGraphConfiguration.GraphBuilder#addVertex(String, GraphVertex, String...)
         */
        public Builder addVertex(String name, GraphVertex vertex, String... inputs) {
            graphBuilder.addVertex(name, vertex, inputs);
            checkFirst(name);
            last = name;
            return this;
        }

        private void checkFirst(String name) {
            if(first == null) {
                first = name;
            }
        }

        /**
         * Build a new OdeVertex
         * @return a new OdeVertex
         */
        public OdeVertex build() {
            return new OdeVertex(graphBuilder
                    .setOutputs(outputName)
                    .addLayer(outputName, new CnnLossLayer(), last)
                    .build(), first, last);
        }

    }
}
