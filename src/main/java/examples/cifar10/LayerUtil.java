package examples.cifar10;

import lombok.Data;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.activations.impl.ActivationReLU;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.schedule.ExponentialSchedule;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.linalg.schedule.ScheduleType;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Utils for creating layers for CIFAR 10
 *
 * @author Christian Skarby
 */
class LayerUtil {

    @Data
    static class EverySecondEpoch implements ISchedule {

        private final ISchedule schedule;

        private EverySecondEpoch(@JsonProperty("schedule") ISchedule schedule) {
            this.schedule = schedule;
        }

        @Override
        public double valueAt(int iteration, int epoch) {
            return schedule.valueAt(iteration, epoch / 2);
        }

        @Override
        public ISchedule clone() {
            return new EverySecondEpoch(schedule);
        }
    }

    /**
     * Initialize a GraphBuilder for CIFAR10 experiment
     * @return a GraphBuilder for CIFAR10 experiment
     */
    public static ComputationGraphConfiguration.GraphBuilder initGraphBuilder(long seed, String... inputs) {
        InputType[] inputTypes = {InputType.convolutional(32,32,3)};
        if(inputs.length == 2) {
            inputTypes = new InputType[] {inputTypes[0], InputType.feedForward(2)};
        }

        return new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.RELU)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new RmsProp.Builder()
                        .learningRateSchedule(
                                new EverySecondEpoch(new ExponentialSchedule(ScheduleType.EPOCH, 0.045, 0.94)))
                        .epsilon(1.0)
                        .rmsDecay(0.9)
                        .build()

                )
                .graphBuilder()
                .setInputTypes(inputTypes)
                .addInputs(inputs);
    }

    /**
     * Create a 1x1 convolution layer
     * @param nrofKernels Number of Kernels
     * @return a {@link Convolution2D}
     */
    static Layer conv1x1(long nrofKernels) {
        return conv1x1(nrofKernels, new ActivationIdentity());
    }

    /**
     * Create a 1x1 convolution layer
     * @param nrofKernels Number of Kernels
     * @param activation Activation function to use
     * @return a {@link Convolution2D}
     */
    static Layer conv1x1(long nrofKernels, IActivation activation) {
        return convAxBSame(nrofKernels, activation, 1,1);
    }

    /**
     * Create a 3x3 convolution layer which is same-padded, i.e. the output feature maps have the same size as the input
     * feature maps
     * @param nrofKernels Number of Kernels
     * @return a {@link Convolution2D}
     */
    static Layer conv3x3Same(long nrofKernels) {
        return conv3x3Same(nrofKernels, new ActivationIdentity());
    }

    /**
     * Create a 3x3 convolution layer which is same-padded, i.e. the output feature maps have the same size as the input
     * feature maps
     * @param nrofKernels Number of Kernels
     * @param activation Activation function to use
     * @return a {@link Convolution2D}
     */
    static Layer conv3x3Same(long nrofKernels, IActivation activation) {
        return convAxBSame(nrofKernels, activation, 3,3);
    }

    /**
     * Create a 1x7 convolution layer which is same-padded, i.e. the output feature maps have the same size as the input
     * feature maps
     * @param nrofKernels Number of Kernels
     * @return a {@link Convolution2D}
     */
    static Layer conv1x7Same(long nrofKernels) {
        return conv1x7Same(nrofKernels, new ActivationIdentity());
    }

    /**
     * Create a 1x7 convolution layer which is same-padded, i.e. the output feature maps have the same size as the input
     * feature maps
     * @param nrofKernels Number of Kernels
     * @param activation Activation function to use
     * @return a {@link Convolution2D}
     */
    static Layer conv1x7Same(long nrofKernels, IActivation activation) {
        return convAxBSame(nrofKernels, activation, 1,7);
    }

    /**
     * Create a 7x1 convolution layer which is same-padded, i.e. the output feature maps have the same size as the input
     * feature maps
     * @param nrofKernels Number of Kernels
     * @return a {@link Convolution2D}
     */
    static Layer conv7x1Same(long nrofKernels) {
        return conv7x1Same(nrofKernels, new ActivationIdentity());
    }

    /**
     * Create a 7x1 convolution layer which is same-padded, i.e. the output feature maps have the same size as the input
     * feature maps
     * @param nrofKernels Number of Kernels
     * @param activation Activation function to use
     * @return a {@link Convolution2D}
     */
    static Layer conv7x1Same(long nrofKernels, IActivation activation) {
        return convAxBSame(nrofKernels, activation, 7,1);
    }

    /**
     * Create a 1x3 convolution layer which is same-padded, i.e. the output feature maps have the same size as the input
     * feature maps
     * @param nrofKernels Number of Kernels
     * @return a {@link Convolution2D}
     */
    static Layer conv1x3Same(long nrofKernels) {
        return conv1x3Same(nrofKernels, new ActivationIdentity());
    }

    /**
     * Create a 1x3 convolution layer which is same-padded, i.e. the output feature maps have the same size as the input
     * feature maps
     * @param nrofKernels Number of Kernels
     * @param activation Activation function to use
     * @return a {@link Convolution2D}
     */
    static Layer conv1x3Same(long nrofKernels, IActivation activation) {
        return convAxBSame(nrofKernels, activation, 1,3);
    }

    /**
     * Create a 3x1 convolution layer which is same-padded, i.e. the output feature maps have the same size as the input
     * feature maps
     * @param nrofKernels Number of Kernels
     * @return a {@link Convolution2D}
     */
    static Layer conv3x1Same(long nrofKernels) {
        return conv3x1Same(nrofKernels, new ActivationIdentity());
    }

    /**
     * Create a 3x1 convolution layer which is same-padded, i.e. the output feature maps have the same size as the input
     * feature maps
     * @param nrofKernels Number of Kernels
     * @param activation Activation function to use
     * @return a {@link Convolution2D}
     */
    static Layer conv3x1Same(long nrofKernels, IActivation activation) {
        return convAxBSame(nrofKernels, activation, 3,1);
    }

    static Layer convAxBSame(long nrofKernels, IActivation activation, int ... kernelSize) {
        return new Convolution2D.Builder(kernelSize)
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
    static Layer conv3x3(long nrofKernels, int... strides) {
        return new Convolution2D.Builder(3, 3)
                .nOut(nrofKernels)
                .stride(strides)
                .activation(new ActivationIdentity())
                .build();
    }

    static Layer maxPool3x3(int ...strides) {
        return new Pooling2D.Builder()
                .poolingType(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(3, 3)
                .stride(strides)
                .build();
    }

    /**
     * Create a {@link BatchNormalization} layer with ReLU activation
     * @return a {@link BatchNormalization}
     */
    static Layer norm() {
        return new BatchNormalization.Builder()
                .activation(new ActivationReLU()).build();
    }

}
