package examples.mnist;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.Convolution2D;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.activations.impl.ActivationReLU;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;

/**
 * Utils for creating layers
 *
 * @author Christian Skarby
 */
class LayerUtil {

    /**
     * Initialize a GraphBuilder for MNIST
     * @return a GraphBuilder for MNIST
     */
    public static ComputationGraphConfiguration.GraphBuilder initGraphBuilder(long seed) {
        return new NeuralNetConfiguration.Builder()
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

    /**
     * Create a 3x3 convolution layer which is same-padded, i.e. the output feature maps have the same size as the input
     * feature maps
     * @param nrofKernels Number of Kernerls
     * @return a {@link Convolution2D}
     */
    static Layer conv3x3Same(int nrofKernels) {
        return conv3x3Same(nrofKernels, new ActivationIdentity());
    }

    /**
     * Create a 3x3 convolution layer which is same-padded, i.e. the output feature maps have the same size as the input
     * feature maps
     * @param nrofKernels Number of Kernels
     * @param activation Activation function to use
     * @return a {@link Convolution2D}
     */
    static Layer conv3x3Same(int nrofKernels, IActivation activation) {
        return new Convolution2D.Builder(3, 3)
                .nOut(nrofKernels)
                .hasBias(false)
                .activation(activation)
                .convolutionMode(ConvolutionMode.Same)
                .build();
    }

    /**
     * Create a 3x3 convolution layer
     * feature maps
     * @param nrofKernels Number of Kernerls
     * @return a {@link Convolution2D}
     */
    static Layer conv3x3(int nrofKernels) {
        return new Convolution2D.Builder(3, 3)
                .nOut(nrofKernels)
                .activation(new ActivationIdentity())
                .build();
    }

    /**
     * Create a 3x3 convolution layer which downsamples the input size by a factor of 2
     * feature maps
     * @param nrofKernels Number of Kernerls
     * @return a {@link Convolution2D}
     */
    static Layer conv1x1DownSample(int nrofKernels) {
        return  new Convolution2D.Builder(1, 1)
                .nOut(nrofKernels)
                .stride(2,2) // 1x1 conv with stride 2? Won't this just remove 50% of information?
                .hasBias(false)
                .activation(new ActivationIdentity())
                .build();
    }

    /**
     * Create a 3x3 convolution layer which downsamples the input size by a factor of 2
     * feature maps
     * @param nrofKernels Number of Kernerls
     * @return a {@link Convolution2D}
     */
    static Layer conv3x3DownSample(int nrofKernels) {
        return new Convolution2D.Builder(3, 3)
                .nOut(nrofKernels)
                .hasBias(false)
                .stride(2,2)
                // Reference implementation uses identity, but performance becomes very bad if used here (?!)
                .activation(new ActivationReLU())
                //.activation(new ActivationIdentity())
                .convolutionMode(ConvolutionMode.Same)
                .build();
    }

    /**
     * Create a 4x4 convolution layer which downsamples the input size by a factor of 2
     * feature maps
     * @param nrofKernels Number of Kernerls
     * @return a {@link Convolution2D}
     */
    static Layer conv4x4DownSample(int nrofKernels) {
        return conv4x4DownSample(nrofKernels, new ActivationIdentity());
    }

    /**
     * Create a 4x4 convolution layer which downsamples the input size by a factor of 2
     * feature maps
     * @param nrofKernels Number of Kernerls
     * @return a {@link Convolution2D}
     */
    static Layer conv4x4DownSample(int nrofKernels, IActivation activation) {
        return new Convolution2D.Builder(4, 4)
                .stride(2, 2)
                .nOut(nrofKernels)
                .padding(1, 1)
                .activation(activation)
                .build();
    }

    /**
     * Create a {@link BatchNormalization} layer
     * @param size size of output
     * @return a {@link BatchNormalization}
     */
    static Layer norm(int size) {
        return new BatchNormalization.Builder()
                .nOut(size)
                .activation(new ActivationReLU()).build();
    }

}
