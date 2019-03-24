package examples.spiral;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.weights.WeightInitUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.config.Adam;

import java.util.Map;

/**
 * Utils for creating layers
 *
 * @author Christian Skarby
 */
class LayerUtil {

    /**
     * Initialize a GraphBuilder for 2D spiral generation
     *
     * @return a GraphBuilder for 2D spiral generation
     */
    public static ComputationGraphConfiguration.GraphBuilder initGraphBuilder(long seed, long nrofSamples) {
        return new NeuralNetConfiguration.Builder()
                .seed(seed)
                // At first glance, Pytorch seems to use RELU_UNIFORM for dense layers by default. However, a combination
                // of magic numbers and odd default values results in the equivalent of UNIFORM. The spiral experiment
                // is stupidly sensitive to hyper parameters and weight (and bias) init is no exception to this.
                .weightInit(WeightInit.UNIFORM)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(0.01))
                .graphBuilder()
                .setInputTypes(InputType.recurrent(2, nrofSamples), InputType.feedForward(nrofSamples));
    }

    /**
     * Initialize biases according to the same strategy as is done for the weights for vertices which have a bias parameter
     *
     * @param graph      Graph to init
     * @param weightInit Method for weight init
     */
    public static void initBiases(ComputationGraph graph, WeightInit weightInit) {

        Map<String, INDArray> paramTable = graph.paramTable(false);
        for (String parName : paramTable.keySet()) {
            if (parName.endsWith(DefaultParamInitializer.BIAS_KEY)) {
                initWeigths(parName, paramTable, weightInit);
            }
        }
    }

    private static void initWeigths(String biasKey, Map<String, INDArray> paramTable, WeightInit weightInit) {
        final String weightKey = biasKey.substring(0, biasKey.length() - DefaultParamInitializer.BIAS_KEY.length()) + DefaultParamInitializer.WEIGHT_KEY;

        final INDArray weight = paramTable.get(weightKey);
        double fanIn = 0;
        if (weight.rank() == 2) {
            fanIn = weight.size(0);
        }

        final INDArray bias = paramTable.get(biasKey);

        WeightInitUtil.initWeights(
                fanIn,
                (double) bias.length(),
                bias.shape(),
                weightInit,
                null,
                bias);

    }
}
