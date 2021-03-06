package ode.vertex.conf;

import lombok.Data;
import lombok.EqualsAndHashCode;
import ode.solve.conf.DormandPrince54Solver;
import ode.vertex.conf.helper.GraphInputOutputFactory;
import ode.vertex.conf.helper.NoTimeInputFactory;
import ode.vertex.conf.helper.OdeHelper;
import ode.vertex.conf.helper.TimeInputFactory;
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
import util.preproc.DuplicateScalarToShape;

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
@EqualsAndHashCode(callSuper = false)
public class OdeVertex extends GraphVertex {

    protected ComputationGraphConfiguration conf;
    protected String firstVertex;
    protected OdeHelperForward odeForwardConf;
    protected OdeHelperBackward odeBackwardConf;
    protected GraphInputOutputFactory graphInputOutputFactory;
    protected GradientViewFactory gradientViewFactory;

    public OdeVertex(
            @JsonProperty("conf") ComputationGraphConfiguration conf,
            @JsonProperty("firstVertex") String firstVertex,
            @JsonProperty("odeForwardConf") OdeHelperForward odeForwardConf,
            @JsonProperty("odeBackwardConf") OdeHelperBackward odeBackwardConf,
            @JsonProperty("graphInputOutputFactory") GraphInputOutputFactory graphInputOutputFactory,
            @JsonProperty("gradientViewFactory") GradientViewFactory gradientViewFactory) {
        this.conf = conf;
        this.firstVertex = firstVertex;
        this.odeForwardConf = odeForwardConf;
        this.odeBackwardConf = odeBackwardConf;
        this.graphInputOutputFactory = graphInputOutputFactory;
        this.gradientViewFactory = gradientViewFactory;
    }

    @Override
    public GraphVertex clone() {
        return new OdeVertex(
                conf.clone(),
                firstVertex,
                odeForwardConf.clone(),
                odeBackwardConf.clone(),
                graphInputOutputFactory.clone(),
                gradientViewFactory.clone());
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

        if (initializeParams && paramsView != null) {
            innerGraph.init(); // This will init parameters using weight initialization
            paramsView.assign(innerGraph.params());
        }

        innerGraph.init(paramsView, false); // This does not update any parameters, just sets them

        final DefaultTrainingConfig trainingConfig = new DefaultTrainingConfig(
                innerGraph,
                name,
                gradientViewFactory.paramNameMapping());

        return new ode.vertex.impl.OdeVertex(
                new ode.vertex.impl.OdeVertex.BaseGraphVertexInputs(graph, name, idx),
                new OdeGraphHelper(
                        odeForwardConf.instantiate(),
                        odeBackwardConf.instantiate(),
                        graphInputOutputFactory,
                        new OdeGraphHelper.CompGraphAsOdeFunction(
                                innerGraph,
                                // Hacky handling for legacy models. To be removed...
                                gradientViewFactory == null ? new GradientViewSelectionFromBlacklisted() : gradientViewFactory)
                ),
                trainingConfig);
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType... vertexInputs) throws InvalidInputTypeException {
        final InputType[] graphInputs = graphInputOutputFactory.getInputType(vertexInputs);
        return odeForwardConf.getOutputType(conf, graphInputs);
    }

    @Override
    public MemoryReport getMemoryReport(InputType... inputTypes) {
        return conf.getMemoryReport(inputTypes);
    }

    /**
     * Builds {@link OdeVertex}es.
     */
    public static class Builder {

        private final ComputationGraphConfiguration.GraphBuilder graphBuilder;
        private final String first;
        private OdeHelperForward odeForwardConf = new FixedStep(new DormandPrince54Solver(), Nd4j.arange(2), true);
        private OdeHelperBackward odeBackwardConf = new FixedStepAdjoint(new DormandPrince54Solver(), Nd4j.arange(2));
        private GraphInputOutputFactory graphInputOutputFactory = new NoTimeInputFactory();
        private GradientViewFactory gradientViewFactory = new GradientViewSelectionFromBlacklisted();

        /**
         * Constructs a Builder for an {@link OdeVertex}
         * @param globalConf Configuration to use for internal graph
         * @param name Name of first layer
         * @param layer First layer in internal graph
         */
        public Builder(NeuralNetConfiguration.Builder globalConf, String name, Layer layer) {
            graphBuilder = globalConf.clone().graphBuilder();
            final String inputName = this.toString() + "_input";
            graphBuilder
                    .addInputs(inputName)
                    .addLayer(name, layer, inputName);
            first = name;
        }

        /**
         * Constructs a Builder for an {@link OdeVertex}
         * @param globalConf Configuration to use for internal graph
         * @param name Name of first vertex
         * @param vertex First vertex in internal graph
         * @param timeAsInput True if current time of the ODE solver shall be input to vertex as well
         */
        public Builder(NeuralNetConfiguration.Builder globalConf,
                       String name,
                       GraphVertex vertex,
                       boolean timeAsInput,
                       String ... otherInputs) {
            graphBuilder = globalConf.clone().graphBuilder();
            final String inputName = this.toString() + "_input";
            first = name;
            graphBuilder.addInputs(inputName);
            if(timeAsInput) {
                addTimeVertex(name, vertex, inputName);
            } else {
                addVertex(name, vertex, inputName);
            }
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
         * Add a layer which takes current time in the ODE solver as input. A {@link DuplicateScalarToShape} is added
         * so that mini batch size is the same as for other layers
         *
         */
        public Builder addTimeLayer(String name, Layer layer) {
            final String[] withTime = setTimeInputs(new String[0]);
            graphBuilder.addLayer(name, layer, new DuplicateScalarToShape(), withTime);
            return this;
        }

        /**
         /**
         * Add a vertex which in addition to the given inputs also takes the current time in the ODE solver as an input.
         * Note that time is a scalar so it is usually required to at least duplicate it to the mini batch size.
         *
         * @see ComputationGraphConfiguration.GraphBuilder#addVertex(String, GraphVertex, String...)
         */
        public Builder addTimeVertex(String name, GraphVertex vertex, String... inputs) {
            final String[] withTime = setTimeInputs(inputs);
            return addVertex(name, vertex, withTime);
        }

        private String[] setTimeInputs(String[] inputs) {
            graphInputOutputFactory(new TimeInputFactory());

            final String timeInputName = this.toString() + "_timeInput";

            if(!graphBuilder.getNetworkInputs().contains(timeInputName)) {
                graphBuilder.addInputs(timeInputName);
            }

            final String[] withTime = new String[inputs.length + 1];
            System.arraycopy(inputs, 0, withTime, 0, inputs.length);
            withTime[inputs.length] = timeInputName;
            return withTime;
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
         * Sets the {@link GradientViewFactory} to use. Typically not set as default should cover all cases.
         *
         * @param gradientViewFactory Factory for gradient views
         * @return the Builder for fluent API
         */
        public Builder gradientViewFactory(GradientViewFactory gradientViewFactory) {
            this.gradientViewFactory = gradientViewFactory;
            return this;
        }

        /**
         * Sets the {@link GraphInputOutputFactory} to use
         *
         * @param graphInputOutputFactory Factory for graph input and output to use. Is typically set automatically.
         * @return the Builder for fluent API
         */
        public Builder graphInputOutputFactory(GraphInputOutputFactory graphInputOutputFactory) {
            this.graphInputOutputFactory = graphInputOutputFactory;
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
                    graphInputOutputFactory,
                    gradientViewFactory);
        }

    }
}
