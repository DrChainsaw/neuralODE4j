package ode.vertex.conf;

import lombok.Data;
import ode.solve.conf.DormandPrince54Solver;
import ode.vertex.conf.helper.OdeHelper;
import ode.vertex.conf.helper.backward.FixedStepAdjoint;
import ode.vertex.conf.helper.backward.OdeHelperBackward;
import ode.vertex.conf.helper.forward.FixedStep;
import ode.vertex.conf.helper.forward.OdeHelperForward;
import ode.vertex.impl.gradview.GradientViewFactory;
import ode.vertex.impl.gradview.GradientViewSelectionFromBlacklisted;
import ode.vertex.impl.helper.OdeGraphHelper;
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
 * Configuration of an ODE block. Contains a {@link ComputationGraphConfiguration} which defines the structure of the
 * learnable function {@code f = z(t)/dt} for which the {@link ode.vertex.impl.OdeVertex} will output an estimate
 * of z(t) for given t(s).
 * <br><br>
 * A {@link Builder} is used to add {@link Layer}s and {@link GraphVertex GraphVertices} to the internal
 * {@link ComputationGraphConfiguration}.
 * <br><br>
 * Note that the internal {@code ComputationGraphConfiguration} is <i>not</i> the same as the "outer"
 * {@code ComputationGraphConfiguration} which houses the OdeVertex itself. This understandably confusing composition
 * comes from the fact that the {@code OdeVertex} needs to operate on an arbitrary graph and I didn't want to
 * reimplement all the routing for doing this. If dl4j had something similar to pytorch's nn.Module I would rather have
 * used that.
 *
 * @author Christian Skarby
 */
@Data
public class OdeVertex extends GraphVertex {

    protected ComputationGraphConfiguration conf;
    protected String firstVertex;
    protected OdeHelperForward odeForwardConf;
    protected OdeHelperBackward odeBackwardConf;
    protected GradientViewFactory gradientViewFactory;

    public OdeVertex(
            @JsonProperty("conf") ComputationGraphConfiguration conf,
            @JsonProperty("firstVertex") String firstVertex,
            @JsonProperty("odeForwardConf") OdeHelperForward odeForwardConf,
            @JsonProperty("odeBackwardConf") OdeHelperBackward odeBackwardConf,
            @JsonProperty("gradientViewFactory") GradientViewFactory gradientViewFactory) {
        this.conf = conf;
        this.firstVertex = firstVertex;
        this.odeForwardConf = odeForwardConf;
        this.odeBackwardConf = odeBackwardConf;
        this.gradientViewFactory = gradientViewFactory;
    }

    @Override
    public GraphVertex clone() {
        return new OdeVertex(conf.clone(), firstVertex, odeForwardConf.clone(), odeBackwardConf.clone(), gradientViewFactory.clone());
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof OdeVertex)) {
            return false;
        }
        final OdeVertex other = (OdeVertex) o;
        return conf.equals(other.conf)
                && firstVertex.equals(other.firstVertex)
                && odeForwardConf.equals(other.odeForwardConf)
                && odeBackwardConf.equals(other.odeBackwardConf)
                && gradientViewFactory.equals(other.gradientViewFactory);
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
        for (GraphVertex vertex : graph.getConfiguration().getVertices().values()) {
            if (vertex != this) {
                atLeastOneLayer |= vertex.numParams(false) > 0;
            }
        }

        final DefaultTrainingConfig trainingConfig = new DefaultTrainingConfig(
                atLeastOneLayer ? graph : innerGraph,
                name);

        return new ode.vertex.impl.OdeVertex(
                new ode.vertex.impl.OdeVertex.BaseGraphVertexInputs(graph, name, idx),
                new OdeGraphHelper(
                        odeForwardConf.instantiate(),
                        odeBackwardConf.instantiate(),
                        new OdeGraphHelper.CompGraphAsOdeFunction(
                                innerGraph,
                                // Hacky handling for legacy models. To be removed...
                                gradientViewFactory == null ? new GradientViewSelectionFromBlacklisted() : gradientViewFactory)
                ),
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

        private final ComputationGraphConfiguration.GraphBuilder graphBuilder = new NeuralNetConfiguration.Builder()
                .graphBuilder();
        private final String first;
        private OdeHelperForward odeForwardConf = new FixedStep(new DormandPrince54Solver(), Nd4j.arange(2), true);
        private OdeHelperBackward odeBackwardConf = new FixedStepAdjoint(new DormandPrince54Solver(), Nd4j.arange(2));
        private GradientViewFactory gradientViewFactory = new GradientViewSelectionFromBlacklisted();

        public Builder(String name, Layer layer) {
            final String inputName = this.toString() + "_input";
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
            return this;
        }

        /**
         * @see ComputationGraphConfiguration.GraphBuilder#addVertex(String, GraphVertex, String...)
         */
        public Builder addVertex(String name, GraphVertex vertex, String... inputs) {
            graphBuilder.addVertex(name, vertex, inputs);
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

        /**
         * Sets the {@link GradientViewFactory} to use
         *
         * @param gradientViewFactory Factory for gradient views
         * @return the Builder for fluent API
         */
        public Builder gradientViewFactory(GradientViewFactory gradientViewFactory) {
            this.gradientViewFactory = gradientViewFactory;
            return this;
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
                    odeForwardConf,
                    odeBackwardConf,
                    gradientViewFactory);
        }

    }
}
