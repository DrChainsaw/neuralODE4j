package examples.spiral;

import ch.qos.logback.classic.Level;
import examples.spiral.listener.PlotDecodedOutput;
import examples.spiral.loss.NormLogLikelihoodLoss;
import ode.solve.conf.DormandPrince54Solver;
import ode.solve.conf.SolverConfig;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import util.listen.training.ZeroGrad;
import util.plot.Plot;
import util.plot.RealTimePlot;

import java.awt.*;
import java.util.Collections;

import static junit.framework.TestCase.assertTrue;

/**
 * Test cases for {@link LatentOdeBlock}, {@link DenseDecoderBlock} and {@link ReconstructionLossBlock} together
 *
 * @author Christian Skarby
 */
public class LatentOdeTest {

    /**
     * Test that the latent ODE can (over)fit to a simple line
     */
    @Test
    public void fitLine() {
        final long nrofTimeSteps = 10;
        final long nrofLatentDims = 20;

        final ComputationGraphConfiguration.GraphBuilder builder = new NeuralNetConfiguration.Builder()
                .seed(666)
                .weightInit(WeightInit.RELU_UNIFORM)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(0.01))
                .graphBuilder()
                .setInputTypes(InputType.feedForward(nrofLatentDims), InputType.feedForward(nrofTimeSteps));
        //.setInputTypes(InputType.recurrent(nrofLatentDims, nrofTimeSteps));

        String next = "z0";
        builder.addInputs(next, "time");
        //builder.addInputs(next);


        next = new LatentOdeBlock(
                "time",
                true,
                nrofLatentDims,
                new DormandPrince54Solver(new SolverConfig(1e-12, 1e-6, 1e-20, 1e2)))
                .add(next, builder);

//        builder.addLayer("rnn", new SimpleRnn.Builder()
//        .nOut(nrofLatentDims)
//        .activation(new ActivationTanH())
//        .build(), next); next = "rnn";

        next = new DenseDecoderBlock(20, 2).add(next, builder);
        final String decoded = next;
        next = new ReconstructionLossBlock(new NormLogLikelihoodLoss(0.3)).add(next, builder);
        builder.setOutputs(next);

        final ComputationGraph graph = new ComputationGraph(builder.build());
        graph.init();

        final INDArray z0 = Nd4j.ones(1, nrofLatentDims);
        //final INDArray z0 = Nd4j.ones(1, nrofLatentDims, nrofTimeSteps);

        final INDArray time = Nd4j.linspace(0, 3, nrofTimeSteps);
        final INDArray label = Nd4j.hstack(time, Nd4j.linspace(0, 9, nrofTimeSteps)).reshape(1, 2, nrofTimeSteps);

        if (!GraphicsEnvironment.isHeadless()
                && !GraphicsEnvironment.getLocalGraphicsEnvironment().isHeadlessInstance()
                && GraphicsEnvironment.getLocalGraphicsEnvironment().getScreenDevices() != null
                && GraphicsEnvironment.getLocalGraphicsEnvironment().getScreenDevices().length > 0) {
            ch.qos.logback.classic.Logger root = (ch.qos.logback.classic.Logger) LoggerFactory.getLogger(Logger.ROOT_LOGGER_NAME);
            root.setLevel(Level.INFO);

            final Plot<Double, Double> linePlot = new RealTimePlot<>("Decoded output", "");
            linePlot.createSeries("Ground truth");
            linePlot.createSeries(decoded);

            new PlotDecodedOutput(linePlot, "Ground truth", 0)
                    .onForwardPass(graph, Collections.singletonMap("Ground truth", label));

            graph.addListeners(new PlotDecodedOutput(linePlot, decoded, 0));
        }

        graph.addListeners(
                new ZeroGrad(),
                new ScoreIterationListener(1));

        final MultiDataSet mds = new MultiDataSet(new INDArray[]{z0, time}, new INDArray[]{label});
        boolean success = false;
        for (int i = 0; i < 300; i++) {
            graph.fit(mds);
            success = graph.score() < 100;
            if (success) break;
        }
        assertTrue("Model failed to train properly! Score after 300 iters: " + graph.score() + "!", success);
    }
}
