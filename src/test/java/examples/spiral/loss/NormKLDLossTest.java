package examples.spiral.loss;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LossLayer;
import org.deeplearning4j.nn.conf.layers.recurrent.SimpleRnn;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.junit.Test;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.activations.impl.ActivationTanH;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.primitives.Pair;

import static junit.framework.TestCase.assertEquals;
import static junit.framework.TestCase.assertTrue;

/**
 * Test cases for {@link NormKLDLoss}
 *
 * @author Christian Skarby
 */
public class NormKLDLossTest {

    /**
     * Test that score and gradient is 0 for a standard gaussian distribution
     */
    @Test
    public void computeGradientAndScoreStandard() {
        final long batchSize = 11;
        final long nrofLatentDimsTimesTwo = 14;
        final INDArray meanAndLogvar = Nd4j.zeros(batchSize, nrofLatentDimsTimesTwo);

        final ILossFunction toTest = new NormKLDLoss();

        final Pair<Double, INDArray> scoreAndGrad = toTest.computeGradientAndScore(meanAndLogvar, meanAndLogvar, new ActivationIdentity(), null, false);

        assertEquals("Score shall be 0! ", 0.0, scoreAndGrad.getFirst(), 1e-10);
        assertEquals("Gradient shall be 0!", meanAndLogvar, scoreAndGrad.getSecond());
    }

    /**
     * Test that gradient is pointing in direction of error
     */
    @Test
    public void gradientMean() {
        final long batchSize = 11;
        final long nrofLatentDimsTimesTwo = 6;
        final INDArray meanAndLogvar = Nd4j.zeros(batchSize, nrofLatentDimsTimesTwo);
        final INDArray expectedMeanAndLogVar = meanAndLogvar.dup();

        meanAndLogvar.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, 2)).assign(Nd4j.create(new double[]{-1.23, 2.34}));
        meanAndLogvar.get(NDArrayIndex.point(2), NDArrayIndex.interval(0, 2)).assign(Nd4j.create(new double[]{3.45, -4.56}));

        final ILossFunction toTest = new NormKLDLoss();

        final INDArray gradient = toTest.computeGradient(expectedMeanAndLogVar, meanAndLogvar, new ActivationIdentity(), null);

        assertEquals("Gradient shall be same as input!", meanAndLogvar, gradient);
    }

    /**
     * Test that gradient is pointing in direction of error
     */
    @Test
    public void gradientLogVar() {
        final long batchSize = 3;
        final long nrofLatentDimsTimesTwo = 4;
        final INDArray meanAndLogvar = Nd4j.zeros(batchSize, nrofLatentDimsTimesTwo);
        final INDArray expectedMeanAndLogVar = meanAndLogvar.dup();

        meanAndLogvar.get(NDArrayIndex.point(0), NDArrayIndex.interval(2, 4)).assign(Nd4j.create(new double[]{-1.23, 2.34}));
        meanAndLogvar.get(NDArrayIndex.point(2), NDArrayIndex.interval(2, 4)).assign(Nd4j.create(new double[]{3.45, -4.56}));

        final ILossFunction toTest = new NormKLDLoss();

        final INDArray gradient = toTest.computeGradient(expectedMeanAndLogVar, meanAndLogvar, new ActivationIdentity(), null);

        assertTrue("Gradient shall be < 0!", gradient.getDouble(0, 2) < 0);
        assertTrue("Gradient shall be > 0!", gradient.getDouble(0, 3) > 0);
        assertEquals("Gradient shall be 0!", 0d, gradient.getDouble(1, 2), 1e-10);
        assertEquals("Gradient shall be 0!", 0d, gradient.getDouble(1, 3), 1e-10);
        assertTrue("Gradient shall be < 0!", gradient.getDouble(2, 2) > 0);
        assertTrue("Gradient shall be > 0!", gradient.getDouble(2, 3) < 0);
    }

    /**
     * Test to teach a small neural network to output zero mean and unit variance
     */
    @Test
    public void learnStandardGaussian() {
        final long batchSize = 3;
        final long nrofInputs = 4;
        final long nrofTimeSteps = 5;

        final DataSet ds = new DataSet(
                Nd4j.linspace(-3, 3, batchSize*nrofInputs*nrofTimeSteps).reshape(batchSize, nrofInputs, nrofTimeSteps),
                Nd4j.zeros(batchSize, nrofInputs));

        final ComputationGraph graph = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .seed(666)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(0.01))
                .graphBuilder()
                .addInputs("input")
                .setInputTypes(InputType.recurrent(nrofInputs, nrofTimeSteps))
                .addLayer("rnn", new SimpleRnn.Builder()
                        .activation(new ActivationTanH())
                        .nOut(20)
                        .build(), "input")
                .addVertex("lastStep", new LastTimeStepVertex("input"), "rnn")
                .addLayer("dnn", new DenseLayer.Builder()
                        .nOut(nrofInputs)
                        .activation(new ActivationIdentity())
                        .build(), "lastStep")

                .addLayer("out", new LossLayer.Builder()
                        .lossFunction(new NormKLDLoss())
                        .build(), "dnn")
                .setOutputs("out")
                .build());
        graph.init();

        boolean success = false;
        for(int i = 0; i < 300; i++) {
            graph.fit(ds);
            success |= graph.score() < 0.001;
            if(success) break;
        }

        assertTrue("Training did not succeed!", success);

    }

}