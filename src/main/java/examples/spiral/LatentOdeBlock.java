package examples.spiral;

import ode.solve.conf.DormandPrince54Solver;
import ode.solve.conf.SolverConfig;
import ode.vertex.conf.OdeVertex;
import ode.vertex.conf.helper.InputStep;
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

    private final String timeName;
    private final boolean interpolateOdeForward;
    private final long nrofLatentDims;

    LatentOdeBlock(String timeName, boolean interpolateOdeForward, long nrofLatentDims) {
        this.timeName = timeName;
        this.interpolateOdeForward = interpolateOdeForward;
        this.nrofLatentDims = nrofLatentDims;
    }

    @Override
    public String add(String prev, ComputationGraphConfiguration.GraphBuilder builder) {
        builder.addVertex("latentOde", new OdeVertex.Builder("fc1",
                new DenseLayer.Builder()
                        .nOut(20)
                        .activation(new ActivationELU()).build())
                .addLayer("fc2", new DenseLayer.Builder()
                        .nOut(20)
                        .activation(new ActivationELU()).build(), "fc1")
                .addLayer("fc3", new DenseLayer.Builder()
                        .nOut(nrofLatentDims)
                        .activation(new ActivationIdentity()).build(), "fc2")
                .odeConf(new InputStep(
                        new DormandPrince54Solver(new SolverConfig(1e-12, 1e-6, 1e-20, 1e2)),
                        1, interpolateOdeForward))
                .build(), prev, timeName);
        return "latentOde";
    }
}
