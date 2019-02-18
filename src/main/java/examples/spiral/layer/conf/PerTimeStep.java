package examples.spiral.layer.conf;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BaseLayer;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * Comprises a sequence of {@link Layer}s which will be activated once per time step in the input. Basically a poor mans
 * version of pytorchs Linear which accepts inputs of any dimension as long as the size of the last dimension is the
 * same as the number of inputs to the layer. This layer only accepts 3D inputs and returns outputs of one higher
 * dimension than the last layer in the sequence (only 3D outputs supported though).
 *
 * @author Christian Skarby
 */
public class PerTimeStep extends FeedForwardLayer {

    private final List<org.deeplearning4j.nn.conf.layers.Layer> layers;

    public PerTimeStep(List<org.deeplearning4j.nn.conf.layers.Layer> layers) {
        this.layers = layers;
        setActivationFn(new ActivationIdentity());
    }

    @Override
    public Layer instantiate(
            NeuralNetConfiguration conf,
            Collection<TrainingListener> trainingListeners,
            int layerIndex,
            INDArray layerParamsView,
            boolean initializeParams) {

        final List<Layer> instLayers = new ArrayList<>();
        int paramCnt = 0;
        for (org.deeplearning4j.nn.conf.layers.Layer layer : layers) {
            final long numPars = layer.initializer().numParams(layer);
            INDArray parView = null;
            if (numPars > 0) {
                parView = layerParamsView.get(NDArrayIndex.point(0), NDArrayIndex.interval(paramCnt, paramCnt + numPars));
            }
            NeuralNetConfiguration layerConf = conf.clone();
            layerConf.setLayer(layer);
            final Layer instLayer = layer.instantiate(layerConf, trainingListeners, layerIndex, parView, initializeParams);
            instLayers.add(instLayer);
            paramCnt += numPars;
        }
        final examples.spiral.layer.impl.PerTimeStep toRet = new examples.spiral.layer.impl.PerTimeStep(conf, instLayers);
        toRet.setParamsViewArray(layerParamsView);

        return toRet;
    }

    @Override
    public ParamInitializer initializer() {
        return new AggParamInitializer(layers);
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        if (inputType.getType() != InputType.Type.RNN) {
            throw new IllegalArgumentException("Unsupported input type: " + inputType);
        }
        InputType.InputTypeRecurrent input = (InputType.InputTypeRecurrent) inputType;

        InputType outputType = InputType.feedForward(input.getSize());
        for (org.deeplearning4j.nn.conf.layers.Layer layer : layers) {
            outputType = layer.getOutputType(layerIndex, outputType);
        }

        if (outputType.getType() != InputType.Type.FF) {
            throw new IllegalArgumentException("Unsupported output type: " + outputType + "!");
        }
        return InputType.recurrent(outputType.arrayElementsPerExample(), input.getTimeSeriesLength());
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        return null;
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        if (inputType.getType() != InputType.Type.RNN) {
            throw new IllegalArgumentException("Unsupported input type: " + inputType);
        }
        InputType.InputTypeRecurrent input = (InputType.InputTypeRecurrent) inputType;
        InputType outputType = InputType.feedForward(input.getSize());

        setNIn(input.getSize());

        for (org.deeplearning4j.nn.conf.layers.Layer layer : layers) {
            layer.setNIn(outputType, override);
            outputType = layer.getOutputType(-1, outputType);
        }
        if (outputType.getType() != InputType.Type.FF) {
            throw new IllegalArgumentException("Unsupported output type: " + outputType + "!");
        }
        setNOut(outputType.arrayElementsPerExample());
    }

    @Override
    public void setWeightInit(WeightInit weightInit) {
        for (org.deeplearning4j.nn.conf.layers.Layer layer : layers) {
            if (layer instanceof BaseLayer) {
                ((BaseLayer) layer).setWeightInit(weightInit);
            }
        }
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        InputType outputType = inputType;
        long numParams = 0;
        long updaterStateSize = 0;
        int trainSizeFixed = 0;
        int trainSizeVariable = 0;

        for (org.deeplearning4j.nn.conf.layers.Layer layer : layers) {

            numParams += layer.initializer().numParams(this);
            if (layer instanceof FeedForwardLayer) {
                updaterStateSize += (int) ((FeedForwardLayer) layer).getIUpdater().stateSize(numParams);
            }

            if (layer.getIDropout() != null) {
                //Assume we dup the input
                trainSizeVariable += inputType.arrayElementsPerExample();
            }

            //Also, during backprop: we do a preOut call -> gives us activations size equal to the output size
            // which is modified in-place by activation function backprop
            // then we have 'epsilonNext' which is equivalent to input size
            outputType = layer.getOutputType(-1, outputType);
            trainSizeVariable += outputType.arrayElementsPerExample();
        }

        return new LayerMemoryReport.Builder(layerName, PerTimeStep.class, inputType, outputType)
                .standardMemory(numParams, updaterStateSize)
                .workingMemory(0, 0, trainSizeFixed, trainSizeVariable) //No additional memory (beyond activations) for inference
                .cacheMemory(MemoryReport.CACHE_MODE_ALL_ZEROS, MemoryReport.CACHE_MODE_ALL_ZEROS) //No caching in DenseLayer
                .build();
    }

    /**
     * Builds {@link PerTimeStep}s
     */
    public static class Builder {

        private final List<org.deeplearning4j.nn.conf.layers.Layer> layers = new ArrayList<>();

        /**
         * Add a {@link org.deeplearning4j.nn.conf.layers.Layer} to the sequence
         *
         * @param layer Layer to add
         * @return the builder for fluent API
         */
        public Builder addLayer(org.deeplearning4j.nn.conf.layers.Layer layer) {
            layers.add(layer);
            return this;
        }

        /**
         * Create a new {@link PerTimeStep} instance
         *
         * @return a new instance
         */
        public PerTimeStep build() {
            return new PerTimeStep(new ArrayList<>(layers));
        }

    }
}
