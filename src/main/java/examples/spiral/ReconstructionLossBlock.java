package examples.spiral;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.layers.RnnLossLayer;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.lossfunctions.ILossFunction;

/**
 * Adds a {@link RnnLossLayer} for the decoder output
 *
 * @author Christian Skarby
 */
class ReconstructionLossBlock implements Block {

    private final ILossFunction loss;

    ReconstructionLossBlock(ILossFunction loss) {
        this.loss = loss;
    }


    @Override
    public String add(String decoderOutput, ComputationGraphConfiguration.GraphBuilder builder) {
        builder.addLayer("reconstruction", new RnnLossLayer.Builder()
                .activation(new ActivationIdentity())
                .lossFunction(loss)
                .build(), decoderOutput);

        return "reconstruction";
    }
}
