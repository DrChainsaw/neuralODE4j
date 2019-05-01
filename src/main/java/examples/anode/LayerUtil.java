package examples.anode;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.learning.config.Adam;

/**
 * Utils for creating layers for ANODE experiments
 *
 * @author Christian Skarby
 */
class LayerUtil {

    /**
     * Initialize a GraphBuilder for 2D spiral generation
     *
     * @return a GraphBuilder for 2D spiral generation
     */
    static ComputationGraphConfiguration.GraphBuilder initGraphBuilder(long nrofInputDims) {
        return new NeuralNetConfiguration.Builder()
                .seed(0)
                .weightInit(WeightInit.UNIFORM)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                // Updater alg not listed in paper?
                .updater(new Adam(1e-3))
                .graphBuilder()
                .setInputTypes(InputType.feedForward(nrofInputDims));
    }
}
