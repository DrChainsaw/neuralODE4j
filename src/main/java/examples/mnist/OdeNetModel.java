package examples.mnist;

import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParametersDelegate;
import ode.solve.api.FirstOrderSolverConf;
import ode.solve.conf.DormandPrince54Solver;
import ode.solve.conf.SolverConfig;
import ode.vertex.conf.OdeVertex;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.Convolution2D;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.activations.impl.ActivationReLU;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import util.listen.step.Mask;
import util.listen.step.StepCounter;

/**
 * OdeNet reference model for. Reimplementation of https://github.com/rtqichen/torchdiffeq/blob/master/examples/odenet_mnist.py
 *
 * @author Christian Skarby
 */
public class OdeNetModel implements ModelFactory {

    private static final Logger log = LoggerFactory.getLogger(OdeNetModel.class);

    @ParametersDelegate
    private StemSelection stemSelection = new StemSelection();

    @Parameter(names = "-nrofKernels", description = "Number of filter kernels in each convolution layer")
    private int nrofKernels = 64;

    @Parameter(names = "-seed", description = "Random seed")
    private long seed = 666;

    private final GraphBuilder builder;

    public OdeNetModel() {
        builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.UNIFORM)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(
                        new MapSchedule.Builder(ScheduleType.EPOCH)
                                .add(0, 0.1)
                                .add(60, 0.01)
                                .add(100, 0.001)
                                .add(140, 0.0001)
                                .build()
                ))
                .graphBuilder()
                .setInputTypes(InputType.convolutionalFlat(28, 28, 1))
                .addInputs("input");
    }

    @Override
    public ComputationGraph create() {
        return create(
                //new NanWatchSolver(
                new DormandPrince54Solver(new SolverConfig(1e-3, 1e-3, 1e-20, 100)));
    }

    ComputationGraph create(FirstOrderSolverConf solver) {

        log.info("Create model");

        solver.addListeners(
                Mask.backward(new StepCounter(20, "Average nrof forward steps: ")),
                Mask.forward(new StepCounter(20, "Average nrof backward steps: "))
        );

        String next = stemSelection.get(nrofKernels).add(builder.getNetworkInputs().get(0), builder);
        next = addOdeBlock(next, solver);
        new Output(nrofKernels).add(next, builder);
        final ComputationGraph graph = new ComputationGraph(builder.build());
        graph.init();
        return graph;
    }

    @Override
    public String name() {
        return "odenet_" + "_" + stemSelection.name();
    }

    private String addOdeBlock(String prev, FirstOrderSolverConf solver) {
        builder
                .addVertex("odeBlock", new OdeVertex.Builder("normFirst",
                        new BatchNormalization.Builder()
                                .nOut(nrofKernels)
                                .activation(new ActivationReLU()).build())
                        .addLayer("convFirst",
                                new Convolution2D.Builder(3, 3)
                                        .nOut(nrofKernels)
                                        .activation(new ActivationIdentity())
                                        .convolutionMode(ConvolutionMode.Same)
                                        .hasBias(false)
                                        .build(), "normFirst")
                        .addLayer("normSecond",
                                new BatchNormalization.Builder()
                                        .nOut(nrofKernels)
                                        .activation(new ActivationReLU()).build(), "convFirst")
                        .addLayer("convSecond",
                                new Convolution2D.Builder(3, 3)
                                        .nOut(nrofKernels)
                                        .activation(new ActivationIdentity())
                                        .convolutionMode(ConvolutionMode.Same)
                                        .hasBias(false)
                                        .build(), "normSecond")
                        .addLayer("normThird",
                                new BatchNormalization.Builder()
                                        .nOut(nrofKernels)
                                        .activation(new ActivationIdentity()).build(), "convSecond")
                        .odeSolver(solver)
                        .build(), prev);
        return "odeBlock";
    }
}
