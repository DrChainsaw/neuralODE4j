package examples.spiral.vertex.impl;

import examples.spiral.vertex.conf.SampleGaussianVertex;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.layers.LossLayer;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.jetbrains.annotations.NotNull;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;

import java.util.stream.LongStream;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link examples.spiral.vertex.conf.SampleGaussianVertex}
 *
 * @author Christian Skarby
 */
public class SampleGaussianVertexTest {

    /**
     * Test that output has the same mean and variance which is input
     */
    @Test
    public void doForward() {
        final long nrofLatentDims = 4;
        final ComputationGraph graph = getGraph(nrofLatentDims, new SampleGaussianVertex(666));

        final int batchSize = 100000;
        final INDArray means = Nd4j.arange(nrofLatentDims);
        final INDArray logVars = Nd4j.arange(nrofLatentDims, 2*nrofLatentDims);
        INDArray output = graph.outputSingle(Nd4j.repeat(means, batchSize), Nd4j.repeat(logVars, batchSize));

        assertArrayEquals("Incorrect mean!", means.toDoubleVector(), output.mean(0).toDoubleVector(), 1e-1);
        assertArrayEquals("Incorrect logvar!", logVars.toDoubleVector(), Transforms.log(output.var(0)).toDoubleVector(), 1e-1);
    }

    /**
     * Test doBackward. LogVar numbers verified in pytorch
     */
    @Test
    public void doBackward() {
        final long nrofLatentDims = 2;
        final ComputationGraph graph = getGraph(nrofLatentDims, new TestSampleGaussianVertex());

        final GraphVertex vertex = graph.getVertex("z");

        // Need to do a forward pass to set inputs and calculate epsilon
        vertex.setInput(0, Nd4j.arange(2*nrofLatentDims), LayerWorkspaceMgr.noWorkspaces());
        vertex.doForward(true, LayerWorkspaceMgr.noWorkspaces());

        vertex.setEpsilon(Nd4j.create(new double[] {1.3, 2.4}));
        final Pair<Gradient, INDArray[]> result = graph.getVertex("z").doBackward(false, LayerWorkspaceMgr.noWorkspaces());

        assertEquals("Incorrect gradient for mean!",
                vertex.getEpsilon(),
                result.getSecond()[0].get(NDArrayIndex.all(), NDArrayIndex.interval(0, nrofLatentDims)));
        assertEquals("Incorrect gradient for log var!",
                Nd4j.create(new double[] {2.6503,   23.1255}),
                result.getSecond()[0].get(NDArrayIndex.all(), NDArrayIndex.interval(nrofLatentDims, 2*nrofLatentDims)));
    }

    @NotNull
    private static ComputationGraph getGraph(long nrofLatentDims, org.deeplearning4j.nn.conf.graph.GraphVertex vertex) {
        final ComputationGraph graph = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs("mean", "logVar")
                .setInputTypes(InputType.feedForward(nrofLatentDims), InputType.feedForward(nrofLatentDims))
                .addVertex("z", vertex, "mean", "logVar") // Note: MergeVertex will be added
                .setOutputs("output")
                .addLayer("output", new LossLayer.Builder().build(), "z")
                .build());
        graph.init();
        return graph;
    }

    private static class TestSampleGaussianVertex extends org.deeplearning4j.nn.conf.graph.GraphVertex {

        private final SampleGaussianVertex helper = new SampleGaussianVertex(666);

        @Override
        public org.deeplearning4j.nn.conf.graph.GraphVertex clone() {
            return null;
        }

        @Override
        public boolean equals(Object o) {
            return false;
        }

        @Override
        public int hashCode() {
            return 0;
        }

        @Override
        public long numParams(boolean backprop) {
            return helper.numParams(backprop);
        }

        @Override
        public int minVertexInputs() {
            return helper.minVertexInputs();
        }

        @Override
        public int maxVertexInputs() {
            return helper.maxVertexInputs();
        }

        @Override
        public GraphVertex instantiate(ComputationGraph graph, String name, int idx, INDArray paramsView, boolean initializeParams) {
            return new examples.spiral.vertex.impl.SampleGaussianVertex(graph, name, idx, shape -> {
                final long sum = LongStream.of(shape).reduce(1, (l1,l2) -> l1*l2);
                return Nd4j.linspace(1.5, 4.3, sum).reshape(shape);
            });
        }

        @Override
        public InputType getOutputType(int layerIndex, InputType... vertexInputs) throws InvalidInputTypeException {
            return helper.getOutputType(layerIndex, vertexInputs);
        }

        @Override
        public MemoryReport getMemoryReport(InputType... inputTypes) {
            return helper.getMemoryReport(inputTypes);
        }
    }
}