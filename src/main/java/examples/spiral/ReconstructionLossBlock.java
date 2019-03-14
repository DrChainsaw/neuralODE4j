package examples.spiral;

import examples.spiral.vertex.conf.LossLayerTransparent;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.lossfunctions.ILossFunction;

/**
 * Adds a {@link LossLayerTransparent} for the decoder output
 *
 * @author Christian Skarby
 */
class ReconstructionLossBlock implements Block {

    private final ILossFunction loss;

    ReconstructionLossBlock(ILossFunction loss) {
        this.loss = loss;
    }


    @Override
    public String add(ComputationGraphConfiguration.GraphBuilder builder, String... decoderOutput) {
        builder.addLayer("reconstruction", new LossLayerTransparent.Builder()
                .activation(new ActivationIdentity())
                .lossFunction(loss)
                .build(), decoderOutput);

        return "reconstruction";
    }
}
