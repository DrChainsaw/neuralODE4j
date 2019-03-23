package examples.spiral;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.learning.config.Adam;

/**
 * Utils for creating layers
 *
 * @author Christian Skarby
 */
class LayerUtil {

    /**
     * Initialize a GraphBuilder for 2D spiral generation
     * @return a GraphBuilder for 2D spiral generation
     */
    public static ComputationGraphConfiguration.GraphBuilder initGraphBuilder(long seed, long nrofSamples) {
        return new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.UNIFORM)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(0.01))
                .graphBuilder()
                .setInputTypes(InputType.recurrent(2, nrofSamples), InputType.feedForward(nrofSamples));
    }
}
