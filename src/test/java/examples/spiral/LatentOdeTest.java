package examples.spiral;

import ch.qos.logback.classic.Level;
import examples.spiral.listener.PlotDecodedOutput;
import examples.spiral.listener.SpiralPlot;
import examples.spiral.loss.NormLogLikelihoodLoss;
import ode.solve.conf.DormandPrince54Solver;
import ode.solve.conf.SolverConfig;
import ode.vertex.conf.helper.InputStep;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.RnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.weights.WeightInitUtil;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import util.listen.training.ZeroGrad;
import util.plot.RealTimePlot;
import util.random.SeededRandomFactory;

import java.awt.*;

import static junit.framework.TestCase.assertEquals;
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
        final long nrofLatentDims = 4;

        //SeededRandomFactory.setNd4jSeed(0);

        final ComputationGraphConfiguration.GraphBuilder builder = new NeuralNetConfiguration.Builder()
                .seed(666)
                .weightInit(WeightInit.UNIFORM)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(0.01))
                .graphBuilder()
                .setInputTypes(InputType.feedForward(nrofLatentDims), InputType.feedForward(nrofTimeSteps));

        String next = "z0";
        builder.addInputs(next, "time");


        next = new LatentOdeBlock(
                20,
                nrofLatentDims,
                new InputStep(
                        new DormandPrince54Solver(new SolverConfig(1e-12, 1e-6, 1e-20, 1e2)),
                        1, true))
                .add(builder, next, "time");

        next = new DenseDecoderBlock(20, 2).add(builder, next);
        final String decoded = next;
        next = new ReconstructionLossBlock(new NormLogLikelihoodLoss(0.3)).add(builder, next);
        builder.setOutputs(next);

        final ComputationGraph graph = new ComputationGraph(builder.build());
        graph.init();

        final INDArray z0 = Nd4j.ones(1, nrofLatentDims);

        final INDArray time = Nd4j.linspace(0, 3, nrofTimeSteps);
        final INDArray label = Nd4j.hstack(time, Nd4j.linspace(0, 9, nrofTimeSteps)).reshape(1, 2, nrofTimeSteps);

        if (!GraphicsEnvironment.isHeadless()
                && !GraphicsEnvironment.getLocalGraphicsEnvironment().isHeadlessInstance()
                && GraphicsEnvironment.getLocalGraphicsEnvironment().getScreenDevices() != null
                && GraphicsEnvironment.getLocalGraphicsEnvironment().getScreenDevices().length > 0) {
            ch.qos.logback.classic.Logger root = (ch.qos.logback.classic.Logger) LoggerFactory.getLogger(Logger.ROOT_LOGGER_NAME);
            root.setLevel(Level.INFO);

            final SpiralPlot linePlot = new SpiralPlot(new RealTimePlot<>("Decoded output", ""));
            linePlot.plot("Ground truth", label.tensorAlongDimension(0, 1, 2));
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
