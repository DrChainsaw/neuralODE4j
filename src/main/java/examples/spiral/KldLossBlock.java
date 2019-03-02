package examples.spiral;

import examples.spiral.loss.NormKLDLoss;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.layers.LossLayer;
import org.nd4j.linalg.activations.impl.ActivationIdentity;

/**
 * Adds a {@link LossLayer} using {@link NormKLDLoss} for the mean and log(var) of the latent variable
 *
 * @author Christian Skarby
 */
class KldLossBlock implements Block {

    @Override
    public String add(String  qz0_meanAndLogvar, ComputationGraphConfiguration.GraphBuilder builder) {
        builder.addLayer("kld", new LossLayer.Builder()
                .activation(new ActivationIdentity())
                .lossFunction(new NormKLDLoss())
                .build(), qz0_meanAndLogvar);
        return "kld";
    }
}
