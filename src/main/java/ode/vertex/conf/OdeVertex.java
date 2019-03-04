package ode.vertex.conf;

import lombok.Data;
import ode.solve.conf.DormandPrince54Solver;
import ode.vertex.conf.helper.OdeHelper;
import ode.vertex.conf.helper.backward.FixedStepAdjoint;
import ode.vertex.conf.helper.backward.OdeHelperBackward;
import ode.vertex.conf.helper.forward.FixedStep;
import ode.vertex.conf.helper.forward.OdeHelperForward;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
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
    protected OdeHelperForward odeForwardConf;
    protected OdeHelperBackward odeBackwardConf;

    public OdeVertex(
            @JsonProperty("conf") ComputationGraphConfiguration conf,
            @JsonProperty("firstVertex") String firstVertex,
            @JsonProperty("lastVertex") String lastVertex,
            @JsonProperty("odeForwardConf") OdeHelperForward odeForwardConf,
            @JsonProperty("odeBackwardConf") OdeHelperBackward odeBackwardConf) {
        this.conf = conf;
        this.firstVertex = firstVertex;
        this.lastVertex = lastVertex;
        this.odeForwardConf = odeForwardConf;
        this.odeBackwardConf = odeBackwardConf;
    }

    @Override
    public GraphVertex clone() {
        return new OdeVertex(conf.clone(), firstVertex, lastVertex, odeForwardConf.clone(), odeBackwardConf.clone());
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof OdeVertex)) {
            return false;
        }
        final OdeVertex other = (OdeVertex) o;
        return conf.equals(other.conf)
                && firstVertex.equals(other.firstVertex)
                && lastVertex.equals(other.lastVertex)
                && odeForwardConf.equals(other.odeForwardConf)
                && odeBackwardConf.equals(other.odeBackwardConf);
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
        return conf.getVertices().get(firstVertex).minVertexInputs() + odeForwardConf.nrofTimeInputs();
    }

    @Override
    public int maxVertexInputs() {
        return conf.getVertices().get(firstVertex).maxVertexInputs() + odeForwardConf.nrofTimeInputs();
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

        // Edge case: "outer" graph has no layers, if so, use the config from one of the layers of the OdeVertex
        boolean atLeastOneLayer = false;
        for(GraphVertex vertex: graph.getConfiguration().getVertices().values()) {
            if(vertex != this) {
                atLeastOneLayer |= vertex.numParams(false) > 0;
            }
        }

        final DefaultTrainingConfig trainingConfig = new DefaultTrainingConfig(
                atLeastOneLayer ? graph : innerGraph,
                name);

        return new ode.vertex.impl.OdeVertex(
                new ode.vertex.impl.OdeVertex.BaseGraphVertexInputs(graph, name, idx),
                innerGraph,
                odeForwardConf.instantiate(),
                odeBackwardConf.instantiate(),
                trainingConfig);
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType... vertexInputs) throws InvalidInputTypeException {
        return odeForwardConf.getOutputType(conf, vertexInputs);
    }

    @Override
    public MemoryReport getMemoryReport(InputType... inputTypes) {
        return conf.getMemoryReport(inputTypes);
    }

    public static class Builder {

        private final String inputName = this.toString() + "_input";
        private final ComputationGraphConfiguration.GraphBuilder graphBuilder = new NeuralNetConfiguration.Builder()
                .graphBuilder();

        private String first;
        private String last;
        private OdeHelperForward odeForwardConf = new FixedStep(new DormandPrince54Solver(), Nd4j.arange(2), true);
        private OdeHelperBackward odeBackwardConf = new FixedStepAdjoint(new DormandPrince54Solver(), Nd4j.arange(2));

        public Builder(String name, Layer layer) {
            graphBuilder
                    .addInputs(inputName)
                    .addLayer(name, layer, inputName);
            first = name;
            last = name;
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
         * Set the {@link OdeHelper} to use
         *
         * @param odeConf ODE configuration
         * @return the Builder for fluent API
         */
        public Builder odeConf(OdeHelper odeConf) {
            odeForward(odeConf.forward());
            return odeBackward(odeConf.backward());
        }

        /**
         * Sets the {@link OdeHelperForward} to use
         *
         * @param odeForwardConf Configuration of forward helper
         * @return the Builder for fluent API
         */
        public Builder odeForward(OdeHelperForward odeForwardConf) {
            this.odeForwardConf = odeForwardConf;
            return this;
        }

        /**
         * Sets the {@link OdeHelperBackward} to use
         *
         * @param odeBackwardConf Configuration of backward helper
         * @return the Builder for fluent API
         */
        public Builder odeBackward(OdeHelperBackward odeBackwardConf) {
            this.odeBackwardConf = odeBackwardConf;
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
                    .allowNoOutput(true)
                    .build(),
                    first,
                    last,
                    odeForwardConf,
                    odeBackwardConf);
        }

    }
}
