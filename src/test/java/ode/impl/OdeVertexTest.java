package ode.impl;

import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.CnnLossLayer;
import org.deeplearning4j.nn.conf.layers.Convolution2D;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Test cases for {@link OdeVertex} (and config of the same)
 */
public class OdeVertexTest {

    /**
     * Smoke test to see that it is possible to do a forward pass
     */
    @Test
    public void doForward() {
        final long nOut = 8;
        final ComputationGraph graph = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs("input")
                .setInputTypes(InputType.convolutional(9, 9, 1))
                .addLayer("0",
                        new Convolution2D.Builder(3, 3)
                                .nOut(nOut)
                                .convolutionMode(ConvolutionMode.Same).build(), "input")
                .addVertex("1", new ode.conf.OdeVertex.Builder("ode0",
                        new BatchNormalization.Builder().build())
                        .addLayer("ode1",
                                new Convolution2D.Builder(3, 3)
                                        .nOut(nOut)
                                        .convolutionMode(ConvolutionMode.Same).build(), "ode0")
                        .build(), "0")
                .addLayer("2", new BatchNormalization.Builder().build(), "1")
                .setOutputs("output")
                .addLayer("output", new CnnLossLayer(), "2")
                .build());

        graph.init();
        graph.output(Nd4j.randn(new long[]{1, 1, 9, 9}));
    }
}