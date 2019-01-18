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
import org.deeplearning4j.nn.conf.layers.Convolution3D;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.ArrayList;
import java.util.List;

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
    protected int timeInputIndex;

    public OdeVertex(
            @JsonProperty("conf") ComputationGraphConfiguration conf,
            @JsonProperty("firstVertex") String firstVertex,
            @JsonProperty("lastVertex") String lastVertex,
            @JsonProperty("odeSolver") FirstOrderSolverConf odeSolver,
            @JsonProperty("timeInputIndex") int timeInputIndex) {
        this.conf = conf;
        this.firstVertex = firstVertex;
        this.lastVertex = lastVertex;
        this.odeSolver = odeSolver;
        this.timeInputIndex = timeInputIndex;
    }

    @Override
    public GraphVertex clone() {
        return new OdeVertex(conf.clone(), firstVertex, lastVertex, odeSolver.clone(), timeInputIndex);
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
                && odeSolver.equals(other.odeSolver)
                && timeInputIndex == other.timeInputIndex;
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
        final int timeOrNot = timeInputIndex == -1 ? 0 : 1;
        return conf.getVertices().get(firstVertex).minVertexInputs() + timeOrNot;
    }

    @Override
    public int maxVertexInputs() {
        final int timeOrNot = timeInputIndex == -1 ? 0 : 1;
        return conf.getVertices().get(firstVertex).maxVertexInputs() + timeOrNot;
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

        // Crappy code! ParamTable might not have deterministic iter order!
        IUpdater updater = innerGraph.getLayer(0).getConfig().getUpdaterByParam(innerGraph.getLayer(0).paramTable().keySet().iterator().next()).clone();
        for(org.deeplearning4j.nn.graph.vertex.GraphVertex vertex: graph.getVertices()) {
            if(vertex != null && vertex.hasLayer()) {
                String parname = vertex.paramTable(false).keySet().iterator().next();
                updater = vertex.getConfig().getUpdaterByParam(parname).clone();
            }
        }

        return new ode.vertex.impl.OdeVertex(
                new ode.vertex.impl.OdeVertex.BaseGraphVertexInputs(graph, name, idx),
                innerGraph,
                odeSolver.instantiate(),
                new DefaultTrainingConfig(name, updater),
                timeInputIndex);
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType... vertexInputs) throws InvalidInputTypeException {
        List<InputType> inputTypeList = new ArrayList<>();
        InputType time = null;
        for (int i = 0; i < vertexInputs.length; i++) {
            if (i != timeInputIndex) { // Never true if timeInputIndex == -1
                inputTypeList.add(vertexInputs[i]);
            } else {
                time = vertexInputs[i];
            }
        }

        InputType outputs = conf.getLayerActivationTypes(inputTypeList.toArray(new InputType[0])).get(lastVertex);

        if(time != null && time.getType() != InputType.Type.FF) {
            throw new IllegalArgumentException("Time must be 1D!");
        }

        return addTimeDim(outputs, time);
    }

    private InputType addTimeDim(InputType type, InputType timeDim) {
        if(timeDim == null) {
            return type;
        }

        switch (type.getType()) {
            case FF: return InputType.recurrent(type.arrayElementsPerExample(), timeDim.arrayElementsPerExample());
            case CNN:
                InputType.InputTypeConvolutional convType = (InputType.InputTypeConvolutional)type;
                return InputType.convolutional3D(Convolution3D.DataFormat.NDHWC,
                        timeDim.arrayElementsPerExample(),
                        convType.getHeight(),
                        convType.getWidth(),
                        convType.getChannels());
            default: throw new UnsupportedOperationException("Input type not supported with time as input!");
        }
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
        private int timeInputIndex = -1;

        private FirstOrderSolverConf odeSolver = new DormandPrince54Solver();

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
         * Sets the {@link FirstOrderSolver} to use
         *
         * @param odeSolver solver instance
         * @return the Builder for fluent API
         */
        public Builder odeSolver(FirstOrderSolverConf odeSolver) {
            this.odeSolver = odeSolver;
            return this;
        }

        /**
         * Indicates that time is given as an input to the vertex. Example:
         * <pre>
         * graphBuilder.addVertex("odeVertex",
         *    new OdeVertex.Builder("0", new DenseLayer.Builder().nOut(4).build())
         *    .timeAsInputIndex(1)
         *    .build(), "someLayer", "time");
         * </pre>
         *
         * @param timeInputIndex input index for time
         * @return the Builder for fluent API
         */
        public Builder timeAsInputIndex(int timeInputIndex) {
            this.timeInputIndex = timeInputIndex;
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
                    odeSolver,
                    timeInputIndex);
        }

    }
}
