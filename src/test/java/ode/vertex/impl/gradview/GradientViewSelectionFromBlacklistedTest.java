package ode.vertex.impl.gradview;

import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.Convolution2D;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.jetbrains.annotations.NotNull;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Map;

import static junit.framework.TestCase.assertEquals;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertNotEquals;

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
        final GradientViewFactory factory = new GradientViewSelectionFromBlacklisted();
        final ParameterGradientView gradView = factory.create(graph);

        assertEquals("Incorrect number of parameter gradients in view!",
                graph.getGradientsViewArray().length() - graph.getLayer("1").getGradientsViewArray().length() / 2,
               gradView.realGradientView().length());

        for(Map.Entry<String, INDArray> nameGradEntry: gradView.allGradientsPerParam().gradientForVariable().entrySet()){
            final Pair<String, String> vertexAndParName = factory.paramNameMapping().reverseMap(nameGradEntry.getKey());
            final long[] expectedShape = graph.getLayer(vertexAndParName.getFirst()).getParam(vertexAndParName.getSecond()).shape();
            assertArrayEquals("Incorrect grad size for " + nameGradEntry.getKey() + "!",
                    expectedShape,
                    nameGradEntry.getValue().shape());
        }
    }

    /**
     * Test that gradients in the black list are not selected
     */
    @Test
    public void createNoBlackList() {
        final ComputationGraph graph = createGraph();
        final GradientViewFactory factory =new GradientViewSelectionFromBlacklisted(new ArrayList<>());
        final ParameterGradientView gradView = factory.create(graph);

        assertEquals("Incorrect number of parameter gradients in view!",
                graph.getGradientsViewArray().length(),
                gradView.realGradientView().length());

        for(Map.Entry<String, INDArray> nameGradEntry: gradView.allGradientsPerParam().gradientForVariable().entrySet()){
            final Pair<String, String> vertexAndParName = factory.paramNameMapping().reverseMap(nameGradEntry.getKey());
            final long[] expectedShape = graph.getLayer(vertexAndParName.getFirst()).getParam(vertexAndParName.getSecond()).shape();
            assertArrayEquals("Incorrect grad size for " + nameGradEntry.getKey() + "!",
                    expectedShape,
                    nameGradEntry.getValue().shape());
        }
    }

    /**
     * Test that a clone is equal to the original
     */
    @Test
    public void clonetest() {
        final GradientViewFactory factory = new GradientViewSelectionFromBlacklisted(Arrays.asList("ff", "gg"));
        assertEquals("Clones shall be equal!", factory, factory.clone());
        assertEquals("Clones shall have equal hashCode!", factory.hashCode(), factory.clone().hashCode());
    }

    /**
     * Test equals
     */
    @Test
    public void equals() {
        assertEquals("Shall be equal!", new GradientViewSelectionFromBlacklisted(), new GradientViewSelectionFromBlacklisted());
        assertEquals("Shall be equal!", new GradientViewSelectionFromBlacklisted(Arrays.asList("aa", "bb")), new GradientViewSelectionFromBlacklisted(Arrays.asList("aa", "bb")));
        assertNotEquals("Shall not be equal!", new GradientViewSelectionFromBlacklisted(Arrays.asList("aa", "bb")), new GradientViewSelectionFromBlacklisted(Arrays.asList("aa", "bb", "cc")));
    }

    /**
     * Test that a {@link GradientViewSelectionFromBlacklisted} can be serialized and then deserialized into the same thing
     */
    @Test
    public void serializeDeserialize() throws IOException {
        final GradientViewFactory factory = new GradientViewSelectionFromBlacklisted(Arrays.asList("qq", "ww"));
        final String json = new ObjectMapper().writeValueAsString(factory);
        final GradientViewFactory deserialized = new ObjectMapper().readValue(json, GradientViewSelectionFromBlacklisted.class);
        assertEquals("Did not deserialize to the same thing!", factory, deserialized);
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