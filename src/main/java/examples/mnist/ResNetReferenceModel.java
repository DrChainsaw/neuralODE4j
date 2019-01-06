package examples.mnist;

import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParametersDelegate;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
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

/**
 * ResNet reference model for comparison. Reimplementation of https://github.com/rtqichen/torchdiffeq/blob/master/examples/odenet_mnist.py
 *
 * @author Christian Skarby
 */
public class ResNetReferenceModel implements ModelFactory {

    private static final Logger log = LoggerFactory.getLogger(ResNetReferenceModel.class);

    @ParametersDelegate
    private StemSelection stemSelection = new StemSelection();

    @Parameter(names = "-nrofResBlocks", description = "Number of residual blocks to use")
    private int nrofResBlocks = 6;

    @Parameter(names = "-nrofKernels", description = "Number of filter kernels in each convolution layer")
    private int nrofKernels = 64;

    @Parameter(names = "-seed", description = "Random seed")
    private long seed = 666;

    private final GraphBuilder builder;

    public ResNetReferenceModel() {
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

        log.info("Create model");

        String next = stemSelection.get(nrofKernels).add(builder.getNetworkInputs().get(0), builder);
        for (int i = 0; i < nrofResBlocks; i++) {
            next = addResBlock(next, i);
        }
        new Output(nrofKernels).add(next, builder);
        final ComputationGraph graph = new ComputationGraph(builder.build());
        graph.init();
        return graph;
    }

    @Override
    public String name() {
        return "resnet_" + nrofResBlocks + "_" + stemSelection.name();
    }


    private String addResBlock(String prev, int cnt) {
        builder
                .addLayer("normFirst_" + cnt,
                        new BatchNormalization.Builder()
                                .nOut(nrofKernels)
                                .activation(new ActivationReLU()).build(), prev)
                .addLayer("convFirst_" + cnt,
                        new Convolution2D.Builder(3, 3)
                                .nOut(nrofKernels)
                                .hasBias(false)
                                .activation(new ActivationIdentity())
                                .convolutionMode(ConvolutionMode.Same)
                                .build(), "normFirst_" + cnt)
                .addLayer("normSecond_" + cnt,
                        new BatchNormalization.Builder()
                                .nOut(nrofKernels)
                                .activation(new ActivationReLU()).build(), "convFirst_" + cnt)
                .addLayer("convSecond_" + cnt,
                        new Convolution2D.Builder(3, 3)
                                .nOut(nrofKernels)
                                .hasBias(false)
                                .activation(new ActivationIdentity())
                                .convolutionMode(ConvolutionMode.Same)
                                .build(), "normSecond_" + cnt)
                .addVertex("add_" + cnt, new ElementWiseVertex(ElementWiseVertex.Op.Add), prev, "convSecond_" + cnt);
        return "add_" + cnt;
    }
}
