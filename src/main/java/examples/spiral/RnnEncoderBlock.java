package examples.spiral;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.recurrent.SimpleRnn;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.activations.impl.ActivationTanH;

/**
 * Simple RRN encode. Structure is different compared to the one used in
 * https://github.com/rtqichen/torchdiffeq/blob/master/examples/latent_ode.py as dl4j does not seem to have the possibility
 * to have a different number of recurrent weights compared to number of outputs. Instead, a {@link DenseLayer} is added
 * after the {@link SimpleRnn} to set output size equal to number of latent dimensions.
 *
 * Furthermore, the RNN in the original implementation concatenates input and previous state of RNN before weighting while
 * SimpleRnn adds input to previous state after input is weighted.
 *
 * @author Christian Skarby
 */
public class RnnEncoderBlock implements Block {

    private final long nrofLatentDims;
    private final long nrofHidden;
    private final String inputName;

    public RnnEncoderBlock(long nrofLatentDims, long nrofHidden, String inputName) {
        this.nrofLatentDims = nrofLatentDims;
        this.nrofHidden = nrofHidden;
        this.inputName = inputName;
    }

    @Override
    public String add(ComputationGraphConfiguration.GraphBuilder builder, String... prev) {
        builder
                // Note that reverse time series input is assumed. ReverseTimeSeriesVertex adds significant processing time
                .addLayer("encRnn", new SimpleRnn.Builder()
                        .nOut(nrofHidden)
                        .activation(new ActivationTanH())
                        .build(),prev)
                .addVertex("encLastStep", new LastTimeStepVertex(inputName), "encRnn")
                .addLayer("encOut", new DenseLayer.Builder()
                .activation(new ActivationIdentity())
                .nOut(2*nrofLatentDims)
                .build(), "encLastStep");

        return "encOut";
    }

}
