package examples.spiral;

import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import examples.spiral.loss.NormLogLikelihoodLoss;
import examples.spiral.vertex.conf.SampleGaussianVertex;
import ode.solve.conf.DormandPrince54Solver;
import ode.solve.conf.SolverConfig;
import ode.vertex.conf.helper.InputStep;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder;
import org.deeplearning4j.nn.conf.graph.SubsetVertex;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;

/**
 * Model used for spiral generation using neural ODE. Equivalent to model used in
 * https://github.com/rtqichen/torchdiffeq/blob/master/examples/latent_ode.py
 *
 * @author Christian Skarby
 */
@Parameters(commandDescription = "Configuration for spiral generating VAE using latent ODE")
class OdeNetModel implements ModelFactory {

    @Parameter(names = "-dontInterpolateOdeForward", description = "Don't use interpolation when solving latent ODE in forward " +
            "direction if set. Default is to use interpolation as this is the method used in original implementation")
    private boolean interpolateOdeForward = true;

    @Parameter(names = "-encoderNrofHidden", description = "Number of hidden units in encoder")
    private long encoderNrofHidden = 25;

    @Parameter(names = "-latentNrofHidden", description = "Number of hidden units in latent ODE function")
    private long latentNrofHidden = 20;

    @Parameter(names = "-decoderNrofHidden", description = "Number of hidden units in decoder")
    private long decoderNrofHidden = 20;

    @Override
    public ComputationGraph create(long nrofSamples, double noiseSigma, long nrofLatentDims) {

        final Block enc = new RnnEncoderBlock(nrofLatentDims, encoderNrofHidden, "spiral");
        final Block dec = new DenseDecoderBlock(decoderNrofHidden, 2);
        final Block ode = new LatentOdeBlock(latentNrofHidden, nrofLatentDims,
                new InputStep(
                        new DormandPrince54Solver(
                                new SolverConfig(1e-12, 1e-6, 1e-20, 1e2)),
                        1, interpolateOdeForward));
        final Block outReconstruction = new ReconstructionLossBlock(new NormLogLikelihoodLoss(noiseSigma));
        final Block outKld = new KldLossBlock();

        final GraphBuilder builder = LayerUtil.initGraphBuilder(666, nrofSamples);
        builder.addInputs("spiral", "time");

        String next = enc.add(builder, "spiral");
        final String qz0_meanAndLogvar = next;

        // Add sampling of a gaussian with the encoded mean and log(var)
        final String qz0_mean = "qz0_mean";
        final String qz0_logvar = "qz0_logvar";
        builder
                //First split prev into mean and log(var) since this is how SampleGaussianVertex assumes input is structured
                .addVertex(qz0_mean, new SubsetVertex(0, (int) nrofLatentDims - 1), next)
                .addVertex(qz0_logvar, new SubsetVertex((int) nrofLatentDims, (int) nrofLatentDims * 2 - 1), next)
                .addVertex("z0", new SampleGaussianVertex(667), qz0_mean, qz0_logvar);
        //.addVertex("z0", new ElementWiseVertex(ElementWiseVertex.Op.Add), qz0_mean, qz0_logvar);

        next = ode.add(builder, "z0", "time"); // Position of "time" is dependent on argument in constructor to InputStep above
        next = dec.add(builder, next);

        // Steps after this is just for ELBO calculation
        String output0 = outReconstruction.add(builder, next);
        String output1 = outKld.add(builder, qz0_meanAndLogvar);

        builder.setOutputs(output0, output1);

        final ComputationGraph graph = new ComputationGraph(builder.build());
        graph.init();

        return graph;
    }

    @Override
    public String name() {
        return "odenet";
    }


    @Override
    public MultiDataSetPreProcessor getPreProcessor(long nrofLatentDims) {
        return new AddKLDLabel(0, 1, nrofLatentDims);
    }
}
