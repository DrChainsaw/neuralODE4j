package examples.spiral;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.graph.PreprocessorVertex;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.activations.impl.ActivationReLU;

/**
 * Simple decoder using {@link DenseLayer}s. Also uses a {@link FeedForwardToRnnPreProcessor} as it assumes 3D input.
 *
 * @author Christian Skarby
 */
public class DenseDecoderBlock implements Block {

    private final long nrofHidden;
    private final long nrofOutputs;

    public DenseDecoderBlock(long nrofHidden, long nrofOutputs) {
        this.nrofHidden = nrofHidden;
        this.nrofOutputs = nrofOutputs;
    }

    @Override
    public String add(ComputationGraphConfiguration.GraphBuilder builder, String... prev) {
        builder
                .addLayer("dec0", new DenseLayer.Builder()
                        .nOut(nrofHidden)
                        .activation(new ActivationReLU())
                        .build(), prev)
                .addLayer("dec1", new DenseLayer.Builder()
                        .nOut(nrofOutputs)
                        .activation(new ActivationIdentity())
                        .build(), "dec0")
                .addVertex("decodedOutput",
                        new PreprocessorVertex(
                                new FeedForwardToRnnPreProcessor()),
                        "dec1");

        return "decodedOutput";
    }
}
