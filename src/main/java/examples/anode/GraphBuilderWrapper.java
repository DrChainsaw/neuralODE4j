package examples.anode;

import ode.vertex.conf.OdeVertex;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.layers.Layer;

/**
 * Generic interface for adding layers to a graph. Useful for having the same methods for adding layers to an {@link ode.vertex.conf.OdeVertex}
 * and to a normal {@link org.deeplearning4j.nn.graph.ComputationGraph}
 */
interface GraphBuilderWrapper {

    /**
     * Add a layer to the wrapped graph builder
     *
     * @param name   Name of layer to add
     * @param layer  Layer to add
     * @param inputs Names of inputs to layer to add
     * @return The {@link GraphBuilderWrapper} for fluent API
     */
    GraphBuilderWrapper addLayer(String name, Layer layer, String... inputs);

    /**
     * Add a vertex to the wrapped graph builder
     *
     * @param name   Name of vertex to add
     * @param vertex  vertex to add
     * @param inputs Names of inputs to vertex to add
     * @return The {@link GraphBuilderWrapper} for fluent API
     */
    GraphBuilderWrapper addVertex(String name, GraphVertex vertex, String... inputs);

    class WrappedGraphBuilder implements GraphBuilderWrapper {
        private final ComputationGraphConfiguration.GraphBuilder graphBuilder;

        WrappedGraphBuilder(ComputationGraphConfiguration.GraphBuilder graphBuilder) {
            this.graphBuilder = graphBuilder;
        }

        @Override
        public GraphBuilderWrapper addLayer(String name, Layer layer, String... inputs) {
            graphBuilder.addLayer(name, layer, inputs);
            return this;
        }

        @Override
        public GraphBuilderWrapper addVertex(String name, GraphVertex vertex, String... inputs) {
            graphBuilder.addVertex(name, vertex, inputs);
            return this;
        }
    }

    class WrappedOdeVertex implements GraphBuilderWrapper {
        private final OdeVertex.Builder builder;

        WrappedOdeVertex(OdeVertex.Builder builder) {
            this.builder = builder;
        }

        @Override
        public GraphBuilderWrapper addLayer(String name, Layer layer, String... inputs) {
            builder.addLayer(name, layer, inputs);
            return this;
        }

        @Override
        public GraphBuilderWrapper addVertex(String name, GraphVertex vertex, String... inputs) {
            builder.addVertex(name, vertex, inputs);
            return this;
        }

    }

    class Wrap implements GraphBuilderWrapper {
        private final GraphBuilderWrapper wrapper;

        Wrap(ComputationGraphConfiguration.GraphBuilder builder) {
            wrapper = new WrappedGraphBuilder(builder);
        }

        Wrap(OdeVertex.Builder builder) {
            wrapper = new WrappedOdeVertex(builder);
        }

        @Override
        public GraphBuilderWrapper addLayer(String name, Layer layer, String... inputs) {
            wrapper.addLayer(name, layer, inputs);
            return this;
        }

        @Override
        public GraphBuilderWrapper addVertex(String name, GraphVertex vertex, String... inputs) {
            wrapper.addVertex(name, vertex, inputs);
            return this;
        }
    }
}
