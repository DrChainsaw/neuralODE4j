package ode.vertex.conf;

import lombok.Data;
import ode.solve.api.FirstOrderSolver;
import ode.solve.api.FirstOrderSolverConf;
import ode.solve.conf.DormandPrince54Solver;
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
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Configuration of an ODE block.
 *
 * @author Christian Skarby
 */
@Data
public class OdeVertex extends GraphVertex {

    protected ComputationGraphConfiguration conf;
    protected String firstVertex;
    protected String lastVertex;
    protected FirstOrderSolverConf odeSolver;

    public OdeVertex(
            @JsonProperty("conf") ComputationGraphConfiguration conf,
            @JsonProperty("firstVertex") String firstVertex,
            @JsonProperty("lastVertex") String lastVertex,
            @JsonProperty("odeSolver") FirstOrderSolverConf odeSolver) {
        this.conf = conf;
        this.firstVertex = firstVertex;
        this.lastVertex = lastVertex;
        this.odeSolver = odeSolver;
    }

    @Override
    public GraphVertex clone() {
        return new OdeVertex(conf.clone(), firstVertex, lastVertex, odeSolver.clone());
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof OdeVertex)) {
            return false;
        }
        final OdeVertex other = (OdeVertex)o;
        return conf.equals(other.conf)
                && firstVertex.equals(other.firstVertex)
                && lastVertex.equals(other.lastVertex)
                && odeSolver.equals(other.odeSolver);
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
    public org.deeplearning4j.nn.graph.vertex.GraphVertex instantiate(
            ComputationGraph graph,
            String name,
            int idx,
            INDArray paramsView,
            boolean initializeParams) {

        final ComputationGraph innerGraph = new ComputationGraph(conf) {

            @Override
            public void init() {
                boolean wasInit = super.initCalled;
                super.init();
                initCalled = wasInit;
            }

            @Override
            public void setBackpropGradientsViewArray(INDArray gradient) {
                flattenedGradients = gradient;
                super.setBackpropGradientsViewArray(gradient);
            }
        };

        if (initializeParams) {
            innerGraph.init(); // This will init parameters using weight initialization
            paramsView.assign(innerGraph.params());
        }

        innerGraph.init(paramsView, false); // This does not update any parameters, just sets them

        return new ode.vertex.impl.OdeVertex(
                graph,
                name,
                idx,
                innerGraph,
                odeSolver.instantiate(),
                new DefaultTrainingConfig(name, graph.getVertices()[1].getConfig().getUpdaterByParam("W").clone()));
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

        private FirstOrderSolverConf odeSolver = new DormandPrince54Solver();

        public Builder(String name, Layer layer) {
            graphBuilder
                    .addInputs(inputName)
                    .addLayer(name, layer, inputName);
            first = name;
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

        /**
         * Sets the {@link FirstOrderSolver} to use
         * @param odeSolver solver instance
         * @return the Builder for fluent API
         */
        public Builder odeSolver(FirstOrderSolverConf odeSolver) {
            this.odeSolver = odeSolver;
            return this;
        }

        private void checkFirst(String name) {
            if (first == null) {
                first = name;
            }
        }

        /**
         * Build a new OdeVertex
         *
         * @return a new OdeVertex
         */
        public OdeVertex build() {
            return new OdeVertex(graphBuilder
                    .setOutputs(outputName)
                    .addLayer(outputName, new CnnLossLayer(), last)
                    .build(), first, last, odeSolver);
        }

    }
}
