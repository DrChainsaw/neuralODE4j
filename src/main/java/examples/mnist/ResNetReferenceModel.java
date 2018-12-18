package examples.mnist;

import com.beust.jcommander.Parameter;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.activations.impl.ActivationReLU;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT;
import org.nd4j.linalg.schedule.ScheduleType;
import org.nd4j.linalg.schedule.StepSchedule;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * ResNet reference model for comparison. Reimplementation of https://github.com/rtqichen/torchdiffeq/blob/master/examples/odenet_mnist.py
 *
 * @author Christian Skärby
 */
public class ResNetReferenceModel {

    private static final Logger log = LoggerFactory.getLogger(ResNetReferenceModel.class);

    @Parameter(names = "-nrofResBlocks", description = "Number of residual blocks to use")
    private int nrofResBlocks = 6;

    @Parameter(names = "-nrofKernels", description = "Number of filter kernels in each convolution layer")
    private int nrofKernels = 64;
    
    private final GraphBuilder builder;

    public ResNetReferenceModel() {
        builder = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.RELU)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(new StepSchedule(ScheduleType.EPOCH, 0.1, 0.1, 60)))
                .graphBuilder()
        .setInputTypes(InputType.convolutional(28,28,1));
    }

    ComputationGraph create() {

        log.info("Create model");

        String next = addStem();
        for(int i = 0; i < nrofResBlocks; i++) {
            next = addResBlock(next, i);
        }
        addOutput(next);
        final ComputationGraph graph = new ComputationGraph(builder.build());
        graph.init();
        return graph;
    }

    private String addStem() {
        builder
                .addInputs("input")
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
                                .stride(2,2)
                                .nOut(nrofKernels)
                                .activation(new ActivationIdentity())
                                .convolutionMode(ConvolutionMode.Same)
                                .build(), "firstNorm")
                .addLayer("secondNorm",
                        new BatchNormalization.Builder()
                                .activation(new ActivationReLU()).build(), "secondConv")
                .addLayer("thirdConv",
                        new Convolution2D.Builder(4, 4)
                                .stride(2,2)
                                .nOut(nrofKernels)
                                .activation(new ActivationIdentity())
                                .convolutionMode(ConvolutionMode.Same)
                                .build(), "secondNorm");

        return "thirdConv";
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
                                .activation(new ActivationIdentity())
                                .convolutionMode(ConvolutionMode.Same)
                                .build(), "normFirst_" + cnt)
                .addLayer("normSecond_" + cnt,
                        new BatchNormalization.Builder()
                                .nOut(nrofKernels)
                                .activation(new ActivationIdentity()).build(), "convFirst_" + cnt)
                .addLayer("convSecond_" + cnt,
                        new Convolution2D.Builder(3, 3)
                                .nOut(nrofKernels)
                                .activation(new ActivationIdentity())
                                .convolutionMode(ConvolutionMode.Same)
                                .build(), "normSecond_" + cnt)
                .addVertex("add_" + cnt, new ElementWiseVertex(ElementWiseVertex.Op.Add), prev, "convSecond_" + cnt);
        return "add_" + cnt;
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
