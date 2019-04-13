package ode.vertex.conf;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.CnnLossLayer;
import org.deeplearning4j.nn.conf.layers.Convolution2D;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/**
 * Test cases for {@link OdeVertex}
 *
 * @author Christian Skarby
 */
public class OdeVertexTest {

    /**
     * Test than an {@link OdeVertex} can be clones
     *
     */
    @Test
    public void cloneTest() {
        final GraphVertex vertex = new OdeVertex.Builder(
                new NeuralNetConfiguration.Builder(), "1", new BatchNormalization.Builder().nOut(3).build())
                .addLayer("2", new ConvolutionLayer.Builder(3, 3).nOut(3).build(), "1")
                .build();

        assertEquals("Clone not equal!", vertex, vertex.clone());
        assertEquals("Hash code of clone not equal!", vertex.hashCode(), vertex.clone().hashCode());

    }

    /**
     * Test than an {@link OdeVertex} can be serialized and deserialized.
     *
     * @throws IOException
     */
    @Test
    public void serializeDeserialize() throws IOException {
        final GraphVertex vertex = new OdeVertex.Builder(
                new NeuralNetConfiguration.Builder(), "1", new BatchNormalization.Builder().nOut(3).build())
                .addLayer("2", new ConvolutionLayer.Builder(3, 3).nOut(3).build(), "1")
                .build();

        final String json = NeuralNetConfiguration.mapper().writeValueAsString(vertex);
        final OdeVertex newVertex = NeuralNetConfiguration.mapper().readValue(json, OdeVertex.class);
        assertEquals("Not same!", vertex, newVertex);
        assertEquals("Not same!", vertex.hashCode(), newVertex.hashCode());
    }

    /**
     * Test that a model with an {@link OdeVertex} can be serialized and deserialized.
     *
     * @throws IOException
     */
    @Test
    public void serializeDeserializeModel() throws IOException {
        final ComputationGraph graph = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .updater(new Sgd())
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .seed(666)
                .graphBuilder()
                .setInputTypes(InputType.convolutional(9, 9, 3))
                .addInputs("input")
                .addLayer("1", new Convolution2D.Builder(3, 3)
                        .convolutionMode(ConvolutionMode.Same)
                        .nOut(6)
                        .build(), "input")
                .addVertex("2",
                        new OdeVertex.Builder(
                                new NeuralNetConfiguration.Builder(), "ode1", new BatchNormalization.Builder().nOut(6).build())
                                .addLayer("ode2", new Convolution2D.Builder(3, 3).convolutionMode(ConvolutionMode.Same)
                                        .nOut(6)
                                        .build(), "ode1")
                                .build(), "1")
                .addLayer("output", new CnnLossLayer(), "2")
                .setOutputs("output")
                .build());
        graph.init();

        final Path baseDir = Paths.get("src", "test", "resources", "OdeVertexTest");
        final String fileName = Paths.get(baseDir.toString(), "testSerializeDeserialize.zip").toString();

        try {

            baseDir.toFile().mkdirs();
            graph.save(new File(fileName), true);
            final ComputationGraph newGraph = ModelSerializer.restoreComputationGraph(new File(fileName), true);

            assertEquals("Config was not restored properly!", graph.getConfiguration(), newGraph.getConfiguration());

            final INDArray input = Nd4j.randn(new long[]{1, 3, 9, 9});
            assertEquals("Output not the same!", graph.outputSingle(input), newGraph.outputSingle(input));

        } catch (IOException e) {
            e.printStackTrace();
            fail("Failed to serialize or deserialize graph!");
        } finally {
            new File(fileName).delete();
            Files.delete(baseDir);
        }
    }

}