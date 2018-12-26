package examples.mnist;

import com.beust.jcommander.Parameter;
import ode.solve.api.FirstOrderSolver;
import ode.solve.commons.FirstOrderSolverAdapter;
import ode.vertex.conf.OdeVertex;
import org.apache.commons.math3.ode.nonstiff.DormandPrince54Integrator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.activations.impl.ActivationReLU;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * OdeNet reference model for. Reimplementation of https://github.com/rtqichen/torchdiffeq/blob/master/examples/odenet_mnist.py
 *
 * @author Christian Skarby
 */
public class OdeNetModel {

    private static final Logger log = LoggerFactory.getLogger(OdeNetModel.class);

    @Parameter(names = "-nrofKernels", description = "Number of filter kernels in each convolution layer")
    private int nrofKernels = 64;

    private final GraphBuilder builder;

    public OdeNetModel() {
        builder = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.RELU)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(
                        new MapSchedule.Builder(ScheduleType.EPOCH)
                                .add(0, 0.01)
                                .add(60, 0.01)
                                .add(100, 0.001)
                                .add(140, 0.0001)
                                .build()
                ))
                .graphBuilder()
                .setInputTypes(InputType.feedForward(28 * 28));
    }

    ComputationGraph create() {
        //return create(new FirstOrderSolverAdapter(new ClassicalRungeKuttaIntegrator(0.5)));
        return create(new FirstOrderSolverAdapter(new DormandPrince54Integrator(
                1e-20, 10d, 1e-1, 1e-2)));
    }

    ComputationGraph create(FirstOrderSolver solver) {

        log.info("Create model");

        String next = addStem();
        next = addOdeBlock(next, solver);
        addOutput(next);
        final ComputationGraph graph = new ComputationGraph(builder.build());
        graph.init();
        return graph;
    }

    private String addStem() {
        builder
                .addInputs("input")
                .inputPreProcessor("firstConv", new FeedForwardToCnnPreProcessor(28, 28))
                .addLayer("firstConv",
                        new Convolution2D.Builder(3, 3)
                                .nOut(nrofKernels)
                                .convolutionMode(ConvolutionMode.Same)
                                .activation(new ActivationIdentity())
                                .build(), "input")
                .addLayer("firstNorm",
                        new BatchNormalization.Builder()
                                .activation(new ActivationReLU()).build(), "firstConv")
                .addLayer("secondConv",
                        new Convolution2D.Builder(4, 4)
                                .stride(2, 2)
                                .nOut(nrofKernels)
                                .activation(new ActivationIdentity())
                                .convolutionMode(ConvolutionMode.Same)
                                .build(), "firstNorm")
                .addLayer("secondNorm",
                        new BatchNormalization.Builder()
                                .activation(new ActivationReLU()).build(), "secondConv")
                .addLayer("thirdConv",
                        new Convolution2D.Builder(4, 4)
                                .stride(2, 2)
                                .nOut(nrofKernels)
                                .activation(new ActivationIdentity())
                                .convolutionMode(ConvolutionMode.Same)
                                .build(), "secondNorm");

        return "thirdConv";
    }

    private String addOdeBlock(String prev, FirstOrderSolver solver) {
        builder
                .addVertex("odeBlock", new OdeVertex.Builder("normFirst_",
                        new BatchNormalization.Builder()
                                .nOut(nrofKernels)
                                .activation(new ActivationReLU()).build())
                .addLayer("convFirst_",
                        new Convolution2D.Builder(3, 3)
                                .nOut(nrofKernels)
                                .activation(new ActivationIdentity())
                                .convolutionMode(ConvolutionMode.Same)
                                .build(), "normFirst_")
                .addLayer("normSecond_",
                        new BatchNormalization.Builder()
                                .nOut(nrofKernels)
                                .activation(new ActivationIdentity()).build(), "convFirst_")
                .addLayer("convSecond_",
                        new Convolution2D.Builder(3, 3)
                                .nOut(nrofKernels)
                                .activation(new ActivationIdentity())
                                .convolutionMode(ConvolutionMode.Same)
                                .build(), "normSecond_")
                        .odeSolver(solver)
                        .build(), prev);
        return "odeBlock";
    }

    private void addOutput(String prev) {
        builder
                .addLayer("normOutput",
                        new BatchNormalization.Builder()
                                .nOut(nrofKernels)
                                .activation(new ActivationReLU()).build(), prev)
                .addLayer("globPool", new GlobalPoolingLayer.Builder().poolingType(PoolingType.AVG).build(), "normOutput")
                .addLayer("output", new OutputLayer.Builder()
                        .nOut(10)
                        .lossFunction(new LossMCXENT())
                        .activation(new ActivationSoftmax()).build(), "globPool")
                .setOutputs("output");
    }

}
