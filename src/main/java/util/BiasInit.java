package util;

import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.distribution.Distributions;
import org.deeplearning4j.nn.conf.layers.BaseLayer;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.weights.WeightInitUtil;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Utility function for initializing bias parameters according to the same strategy as used for weight parameters for
 * each layer. Uses nIn and nOut as fan in and out so will not yield "correct" results for e.g. Conv layers.
 *
 * @author Christian Skarby
 */
public class BiasInit {

    /**
     * Initialize biases according to the same strategy as is done for the weights for vertices which have a bias parameter
     * @param graph Graph to init
     */
    public static void initBiases(ComputationGraph graph) {

        for(GraphVertex vertex: graph.getVertices()) {
            if(vertex.getConfig() instanceof BaseLayer) {
                BaseLayer layer =((BaseLayer) vertex.getConfig());
                INDArray bias = vertex.paramTable(false).get(DefaultParamInitializer.BIAS_KEY);
                initWeigths(bias, layer);
            }
        }
    }

    private static void initWeigths(INDArray bias, BaseLayer layer) {
        final WeightInit weightInit = layer.getWeightInit();
        final Distribution distribution = layer.getDist();
        final long fanIn = ((FeedForwardLayer)layer).getNIn();
        if(bias != null) {
            WeightInitUtil.initWeights(
                    (double)fanIn,
                    (double) bias.length(),
                    bias.shape(),
                    weightInit,
                    Distributions.createDistribution(distribution),
                    bias);
        }
    }
}
