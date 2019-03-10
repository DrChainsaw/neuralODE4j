package ode.vertex.impl.gradview;

import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.Convolution2D;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.jetbrains.annotations.NotNull;
import org.junit.Test;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

import static junit.framework.TestCase.assertEquals;
import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.assertFalse;

/**
 * Test cases for {@link GradientViewSelectionFromBlacklisted}
 *
 * @author Christian Skarby
 */
public class GradientViewSelectionFromBlacklistedTest {

    /**
     * Test that gradients in the black list are not selected
     */
    @Test
    public void createWithBlacklisted() {
        final ComputationGraph graph = createGraph();

        assertEquals("Incorrect number of parameter gradients in view!",
                graph.getGradientsViewArray().length() - graph.getLayer("1").getGradientsViewArray().length() / 2,
                new GradientViewSelectionFromBlacklisted().create(graph).length());
    }

    /**
     * Test that gradients in the black list are not selected
     */
    @Test
    public void createNoBlackList() {
        final ComputationGraph graph = createGraph();

        assertEquals("Incorrect number of parameter gradients in view!",
                graph.getGradientsViewArray().length(),
                new GradientViewSelectionFromBlacklisted(new ArrayList<>()).create(graph).length());
    }

    /**
     * Test that a clone is equal to the original
     */
    @Test
    public void clonetest() {
        final GradientViewFactory factory = new GradientViewSelectionFromBlacklisted(Arrays.asList("ff", "gg"));
        assertTrue("Clones shall be equal!" , factory.equals(factory.clone()));
    }

    /**
     * Test equals
     */
    @Test
    public void equals() {
        assertTrue("Shall be equal!", new GradientViewSelectionFromBlacklisted().equals(new GradientViewSelectionFromBlacklisted()));
        assertTrue("Shall be equal!",
                new GradientViewSelectionFromBlacklisted(Arrays.asList("aa", "bb")).equals(
                        new GradientViewSelectionFromBlacklisted(Arrays.asList("aa", "bb"))));
        assertFalse("Shall not be equal!",
                new GradientViewSelectionFromBlacklisted(Arrays.asList("aa", "bb")).equals(
                        new GradientViewSelectionFromBlacklisted(Arrays.asList("aa", "bb", "cc"))));
    }

    /**
     * Test that a {@link GradientViewSelectionFromBlacklisted} can be serialized and then deserialized into the same thing
     */
    @Test
    public void serializeDeserialize() throws IOException {
        final GradientViewFactory factory = new GradientViewSelectionFromBlacklisted(Arrays.asList("qq", "ww"));
        final String json = new ObjectMapper().writeValueAsString(factory);
        final GradientViewFactory deserialized = new ObjectMapper().readValue(json, GradientViewSelectionFromBlacklisted.class);
        assertTrue("Did not deserialize to the same thing!", factory.equals(deserialized));
    }

    @NotNull
    ComputationGraph createGraph() {
        final ComputationGraph graph = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs("input")
                .addLayer("0", new Convolution2D.Builder().convolutionMode(ConvolutionMode.Same).nOut(3).build(), "input")
                .addLayer("1", new BatchNormalization.Builder().nOut(3).build(), "0")
                .addLayer("2", new Convolution2D.Builder().convolutionMode(ConvolutionMode.Same).nOut(3).build(), "1")
                .setOutputs("2")
                .setInputTypes(InputType.convolutional(5, 5, 3))
                .build());
        graph.init();
        graph.initGradientsView();
        return graph;
    }
}