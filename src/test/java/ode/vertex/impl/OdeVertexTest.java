package ode.vertex.impl;

import ode.solve.conf.DormandPrince54Solver;
import ode.vertex.conf.ShapeMatchVertex;
import ode.vertex.conf.helper.InputStep;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.graph.ScaleVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.junit.Test;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.activations.impl.ActivationReLU;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;

import static org.junit.Assert.assertNotEquals;

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
                .addVertex("1", new ode.vertex.conf.OdeVertex.Builder
                        (new NeuralNetConfiguration.Builder(), "ode0", new BatchNormalization.Builder().build())
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
        INDArray output = graph.outputSingle(Nd4j.randn(new long[]{1, 1, 9, 9}));
        assertNotEquals("Expected non-zero output!", 0, output.sumNumber());
    }

    /**
     * Test that an exception is thrown when input and output to ODE vertex does not match
     */
    @Test(expected = InvalidInputTypeException.class)
    public void inputSizeNotEqualToOutputSize() {
        final long nrofInputs = 3;
        final long nrofOdeOutputs = nrofInputs + 1; // Not allowed!!
        new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs("input")
                .setInputTypes(InputType.feedForward(nrofInputs))
                .addVertex("ode", new ode.vertex.conf.OdeVertex.Builder
                        (new NeuralNetConfiguration.Builder(), "func", new DenseLayer.Builder()
                                .nOut(nrofOdeOutputs)
                                .build())
                        .build(), "input")
                .setOutputs("output")
                .addLayer("output", new LossLayer(), "ode")
                .build(); // Shall fail
    }

    /**
     * Smoke test to see that it is possible to fit
     */
    @Test
    public void fit() {
        final long nOut = 8;
        final ComputationGraph graph = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .updater(new Sgd(0.1))
                .graphBuilder()
                .addInputs("input")
                .setInputTypes(InputType.convolutional(9, 9, 1))
                .addLayer("0",
                        new Convolution2D.Builder(3, 3)
                                .nOut(nOut)
                                .activation(new ActivationIdentity())
                                .convolutionMode(ConvolutionMode.Same).build(), "input")
                .addLayer("bn", new BatchNormalization.Builder()
                        .activation(new ActivationReLU())
                        .build(), "0")
                .addVertex("1", new ode.vertex.conf.OdeVertex.Builder(
                        new NeuralNetConfiguration.Builder(), "ode0", new BatchNormalization.Builder()
                        .activation(new ActivationReLU())
                        .build())
                        .addLayer("ode1",
                                new Convolution2D.Builder(3, 3)
                                        .nOut(nOut)
                                        .activation(new ActivationIdentity())
                                        .convolutionMode(ConvolutionMode.Same).build(), "ode0")
                        .build(), "bn")
                .addLayer("2", new BatchNormalization.Builder()
                        .activation(new ActivationReLU())
                        .build(), "1")
                .addLayer("gp", new GlobalPoolingLayer.Builder().build(), "2")
                .setOutputs("output")
                .addLayer("output", new OutputLayer.Builder().nOut(3).build(), "gp")
                .build());

        graph.init();
        final INDArray before = graph.getVertex("1").params().dup();
        graph.fit(new DataSet(Nd4j.randn(new long[]{1, 1, 9, 9}), Nd4j.create(new double[]{0, 1, 0})));
        assertNotEquals("Expected parameters to be updated!", before, graph.getVertex("1").params().dup());
    }

    /**
     * Smoke test to see that it is possible to fit when time is one of the inputs
     */
    @Test
    public void fitWithTimeAsInput() {
        final long nOut = 8;
        final long nrofTimeSteps = 10;
        final ComputationGraph graph = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs("input", "time")
                .setInputTypes(InputType.feedForward(5), InputType.feedForward(nrofTimeSteps))
                .addLayer("0", new DenseLayer.Builder().nOut(nOut).build(), "input")
                .addVertex("odeVertex", new ode.vertex.conf.OdeVertex.Builder(
                        new NeuralNetConfiguration.Builder(), "ode0", new DenseLayer.Builder().nOut(nOut).build())
                        .odeConf(new InputStep(new DormandPrince54Solver(), 1))
                        .build(), "0", "time")
                .setOutputs("output")
                .addLayer("output", new RnnOutputLayer.Builder().nOut(3).build(), "odeVertex")
                .build());

        graph.init();

        final INDArray before = graph.getVertex("odeVertex").params().dup();
        final int batchSize = 3;
        graph.fit(new MultiDataSet(
                new INDArray[]{Nd4j.randn(new long[]{batchSize, 5}), Nd4j.linspace(0, 2, nrofTimeSteps)},
                new INDArray[]{Nd4j.repeat(Nd4j.create(new double[]{0, 1, 0}).transposei(), batchSize * (int) nrofTimeSteps)}));
        assertNotEquals("Expected parameters to be updated!", before, graph.getVertex("odeVertex").params().dup());
    }

    /**
     * Smoke test to see that it is possible to fit when time is one of the inputs
     */
    @Test
    public void fitWithFunctionOfTime() {
        final long nOut = 8;
        final long nrofTimeSteps = 10;
        final ComputationGraph graph = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs("input", "time")
                .setInputTypes(InputType.feedForward(5), InputType.feedForward(nrofTimeSteps))
                .addLayer("0", new DenseLayer.Builder().nOut(nOut).build(), "input")

                .addVertex("odeVertex",
                        new ode.vertex.conf.OdeVertex.Builder(new NeuralNetConfiguration.Builder(),
                                "ode0", new DenseLayer.Builder().nOut(nOut / 2).build())

                                .addTimeLayer("ode1", new DenseLayer.Builder().nOut(nOut / 2).build())

                                .addLayer("ode3", new DenseLayer.Builder().nOut(nOut).build(), "ode0", "ode1")
                                .odeConf(new InputStep(new DormandPrince54Solver(), 1))
                                .build(), "0", "time")
                .setOutputs("output")
                .addLayer("output", new RnnOutputLayer.Builder().nOut(3).build(), "odeVertex")
                .build());

        graph.init();

        final INDArray before = graph.getVertex("odeVertex").params().dup();
        final int batchSize = 5;
        graph.fit(new MultiDataSet(
                new INDArray[]{Nd4j.randn(new long[]{batchSize, 5}), Nd4j.linspace(0, 2, nrofTimeSteps)},
                new INDArray[]{Nd4j.repeat(Nd4j.create(new double[]{0, 1, 0}).transposei(), batchSize * (int) nrofTimeSteps)}));
        assertNotEquals("Expected parameters to be updated!", before, graph.getVertex("odeVertex").params().dup());
    }

    /**
     * Smoke test to see that it is possible to fit when the first vertex of the internal graph is a graph vertex
     */
    @Test
    public void fitWithFirstVertex() {
        final long nOut = 8;
        final ComputationGraph graph = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs("input")
                .setInputTypes(InputType.feedForward(5))
                .addLayer("0", new DenseLayer.Builder().nOut(nOut).build(), "input")

                .addVertex("odeVertex",
                        new ode.vertex.conf.OdeVertex.Builder(new NeuralNetConfiguration.Builder(),
                                "ode0", new ScaleVertex(0.1), false)

                                .addLayer("ode1", new DenseLayer.Builder().nOut(nOut).build(), "ode0")

                                .build(), "0")
                .setOutputs("output")
                .addLayer("output", new RnnOutputLayer.Builder().nOut(3).build(), "odeVertex")
                .build());

        graph.init();

        final INDArray before = graph.getVertex("odeVertex").params().dup();
        final int batchSize = 5;
        graph.fit(new MultiDataSet(
                new INDArray[]{Nd4j.randn(new long[]{batchSize, 5})},
                new INDArray[]{Nd4j.repeat(Nd4j.create(new double[]{0, 1, 0}).transposei(), batchSize)}));
        assertNotEquals("Expected parameters to be updated!", before, graph.getVertex("odeVertex").params().dup());
    }

    /**
     * Smoke test to see that it is possible to fit when the first vertex of the internal graph is a graph vertex
     */
    @Test
    public void fitWithFirstTimeVertex() {
        final long nOut = 8;
        final ComputationGraph graph = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs("input")
                .setInputTypes(InputType.feedForward(5))
                .addLayer("0", new DenseLayer.Builder().nOut(nOut).build(), "input")

                .addVertex("odeVertex",
                        new ode.vertex.conf.OdeVertex.Builder(new NeuralNetConfiguration.Builder(),
                                "ode0", new ShapeMatchVertex(new MergeVertex()), true)

                                .addLayer("ode1", new DenseLayer.Builder().nOut(nOut).build(), "ode0")

                                .build(), "0")
                .setOutputs("output")
                .addLayer("output", new RnnOutputLayer.Builder().nOut(3).build(), "odeVertex")
                .build());

        graph.init();

        final INDArray before = graph.getVertex("odeVertex").params().dup();
        final int batchSize = 5;
        graph.fit(new MultiDataSet(
                new INDArray[]{Nd4j.randn(new long[]{batchSize, 5})},
                new INDArray[]{Nd4j.repeat(Nd4j.create(new double[]{0, 1, 0}).transposei(), batchSize)}));
        assertNotEquals("Expected parameters to be updated!", before, graph.getVertex("odeVertex").params().dup());
    }
}