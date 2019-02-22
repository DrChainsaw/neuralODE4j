package examples.spiral;

import examples.spiral.vertex.conf.ReverseTimeAsBatch;
import examples.spiral.vertex.conf.TimeAsBatch;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.activations.impl.ActivationReLU;

/**
 * Simple decoder using {@link DenseLayer}s. Also uses a {@link TimeAsBatch} in order to process 3D input as 2D.
 *
 * @author Christian Skarby
 */
public class DenseDecoderBlock implements Block {

    private final long nrofHidden;
    private final long nrofTimeSteps;
    private final long nrofOutputs;

    public DenseDecoderBlock(long nrofHidden, long nrofTimeSteps, long nrofOutputs) {
        this.nrofHidden = nrofHidden;
        this.nrofTimeSteps = nrofTimeSteps;
        this.nrofOutputs = nrofOutputs;
    }

    @Override
    public String add(String prev, ComputationGraphConfiguration.GraphBuilder builder) {
        // TODO: Replace with RnnToFF preprocessor (added automatically)
        builder.addVertex("timeAsBatch", new TimeAsBatch(), prev)
                .addLayer("dec0", new DenseLayer.Builder()
                        .nOut(nrofHidden)
                        .activation(new ActivationReLU())
                        .build(), "timeAsBatch")
                .addLayer("dec1", new DenseLayer.Builder()
                        .nOut(nrofOutputs)
                        .activation(new ActivationIdentity())
                        .build(), "dec0")
                // TODO: Replace with FfToRnn preprocessor
                .addVertex("decodedOutput", new ReverseTimeAsBatch(nrofTimeSteps), "dec1");

        return "decodedOutput";
    }
}
