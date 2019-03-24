package examples.spiral;

import ode.vertex.conf.OdeVertex;
import ode.vertex.conf.helper.OdeHelper;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.nd4j.linalg.activations.impl.ActivationELU;
import org.nd4j.linalg.activations.impl.ActivationIdentity;

/**
 * {@link Block} which uses an {@link OdeVertex} to calculate a latent variable z(t) from z(0) and t
 *
 * @author Christian Skarby
 */
class LatentOdeBlock implements Block {

    private final long nrofHidden;
    private final long nrofLatentDims;
    private final OdeHelper solverConf;


    LatentOdeBlock(long nrofHidden, long nrofLatentDims, OdeHelper solverConf) {
        this.nrofHidden = nrofHidden;
        this.nrofLatentDims = nrofLatentDims;
        this.solverConf = solverConf;
    }

    @Override
    public String add(ComputationGraphConfiguration.GraphBuilder builder, String... prev) {
        builder.addVertex("latentOde", new OdeVertex.Builder(
                builder.getGlobalConfiguration(),
                "fc1",
                new DenseLayer.Builder()
                        .nIn(nrofLatentDims) // Fail fast if previous layer is incorrect
                        .nOut(nrofHidden)
                        .activation(new ActivationELU()).build())
                .addLayer("fc2", new DenseLayer.Builder()
                        .nOut(nrofHidden)
                        .activation(new ActivationELU()).build(), "fc1")
                .addLayer("fc3", new DenseLayer.Builder()
                        .nOut(nrofLatentDims)
                        .activation(new ActivationIdentity()).build(), "fc2")
                .odeConf(solverConf)
                .build(), prev);
        return "latentOde";
    }
}
