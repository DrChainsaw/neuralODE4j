package examples.spiral.vertex.conf;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.RnnLossLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.jetbrains.annotations.NotNull;
import org.junit.Test;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.impl.LossMSE;

import static junit.framework.TestCase.assertEquals;
import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.assertArrayEquals;

/**
 * Test cases for {@link TimeAsBatch} and {@link ReverseTimeAsBatch}
 *
 * @author Christian Skarby
 */
public class TimeAsBatchTest {

    /**
     * Test activation through two layers. First layer multiplies input by 2 and second layer multiplies input by 3. Expected
     * outcome is thus that input has been multiplied by a factor of 6.
     */
    @Test
    public void doForward() {
        final long nrofInputs = 4;
        final long nrofTimeSteps = 10;
        final ComputationGraph graph = createGraph(nrofInputs, nrofTimeSteps);
        graph.getLayer("0").params().assign(Nd4j.eye(nrofInputs).reshape(1, nrofInputs * nrofInputs)).muli(2);
        graph.getLayer("1").params().assign(Nd4j.eye(nrofInputs).reshape(1, nrofInputs * nrofInputs)).muli(3);

        final long batchSize = 3;
        final INDArray input = Nd4j.arange(batchSize * nrofInputs * nrofTimeSteps).reshape(batchSize, nrofInputs, nrofTimeSteps);
        final INDArray expected = input.mul(6);

        final INDArray actual = graph.outputSingle(input);

        assertEquals("Incorrect output!", expected, actual);
    }

    /**
     * Smoke test for doBackward.
     */
    @Test
    public void doBackward() {
        final long nrofInputs = 5;
        final long nrofTimeSteps = 7;
        final ComputationGraph graph = createGraph(nrofInputs, nrofTimeSteps);

        final long batchSize = 11;
        final INDArray input = Nd4j.arange(batchSize * nrofInputs * nrofTimeSteps).reshape(batchSize, nrofInputs, nrofTimeSteps);


        graph.fit(new DataSet(input, input.mul(3)));

        assertTrue("Incorrect output!", graph.getGradientsViewArray().aminNumber().doubleValue() > 0);
    }

    /**
     * Test that a graph using {@link TimeAsBatch} and {@link ReverseTimeAsBatch} can learn that output is sum of inputs
     */
    @Test
    public void learnSum() {
        final int nrofInputs = 2;
        final int nrofTimeSteps = 3;
        final ComputationGraph graph = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.XAVIER)
                .seed(666)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(0.2))
                .graphBuilder()
                .setInputTypes(InputType.recurrent(nrofInputs, nrofTimeSteps))
                .addInputs("input")
                .setOutputs("output")
                .addVertex("timeAsBatch", new TimeAsBatch(), "input")
                .addLayer("0", new DenseLayer.Builder()
                        .nOut(1)
                        .hasBias(false)
                        .activation(new ActivationIdentity())
                        .build(), "timeAsBatch")
                .addVertex("reverseTimeAsBatch", new ReverseTimeAsBatch(nrofTimeSteps), "0")
                .addLayer("output", new RnnLossLayer.Builder()
                        .lossFunction(new LossMSE())
                        .activation(new ActivationIdentity()).build(), "reverseTimeAsBatch")
                .build());
        graph.init();

        final long batchSize = 5;
        for (int i = 0; i < 5000; i++) {
            final INDArray input = Nd4j.linspace(i % 10, i % 10 + 1, nrofInputs * nrofTimeSteps * batchSize).reshape(batchSize, nrofInputs, nrofTimeSteps);
            final INDArray output = input.sum(1).reshape(batchSize, 1, nrofTimeSteps);
            graph.fit(new DataSet(input, output));
        }
        final INDArray input = Nd4j.linspace(3, 7, nrofInputs * nrofTimeSteps).reshape(1, nrofInputs, nrofTimeSteps);
        final INDArray expected = input.sum(1).reshape(1, 1, nrofTimeSteps);

        assertArrayEquals("incorrect output!",
                expected.reshape(nrofTimeSteps).toDoubleVector(),
                graph.outputSingle(input).reshape(nrofTimeSteps).toDoubleVector()
                , 1e-1);
    }

    @NotNull
    private static ComputationGraph createGraph(long nrofInputs, long nrofTimeSteps) {
        final ComputationGraph graph = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.XAVIER)
                .graphBuilder()
                .setInputTypes(InputType.recurrent(nrofInputs, nrofTimeSteps))
                .addInputs("input")
                .setOutputs("output")
                .addVertex("timeAsBatch", new TimeAsBatch(), "input")
                .addLayer("0", new DenseLayer.Builder()
                        .nOut(nrofInputs)
                        .hasBias(false)
                        .activation(new ActivationIdentity())
                        .build(), "timeAsBatch")
                .addLayer("1", new DenseLayer.Builder()
                        .nOut(nrofInputs)
                        .hasBias(false)
                        .activation(new ActivationIdentity()).build(), "0")
                .addVertex("reverseTimeAsBatch", new ReverseTimeAsBatch(nrofTimeSteps), "1")
                .addLayer("output", new RnnLossLayer.Builder().activation(new ActivationIdentity()).build(), "reverseTimeAsBatch")
                .build());
        graph.init();
        return graph;
    }

}