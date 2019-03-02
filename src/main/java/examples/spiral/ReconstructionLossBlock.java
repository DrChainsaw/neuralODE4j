package examples.spiral;

import examples.spiral.loss.NormLogLikelihoodLoss;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.layers.RnnLossLayer;
import org.nd4j.linalg.activations.impl.ActivationIdentity;

/**
 * Adds a {@link RnnLossLayer} using {@link NormLogLikelihoodLoss} for the decoder output
 *
 * @author Christian Skarby
 */
class ReconstructionLossBlock implements Block {

    private final double noiseSigma;

    ReconstructionLossBlock(double noiseSigma) {
        this.noiseSigma = noiseSigma;
    }

    @Override
    public String add(String decoderOutput, ComputationGraphConfiguration.GraphBuilder builder) {
        builder.addLayer("reconstruction", new RnnLossLayer.Builder()
                .activation(new ActivationIdentity())
                .lossFunction(new NormLogLikelihoodLoss(noiseSigma))
                .build(), decoderOutput);

        return "reconstruction";
    }
}
