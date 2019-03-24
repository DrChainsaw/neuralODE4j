package examples.spiral;

import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import examples.spiral.loss.NormLogLikelihoodLoss;
import examples.spiral.vertex.conf.SampleGaussianVertex;
import ode.solve.conf.DormandPrince54Solver;
import ode.solve.conf.SolverConfig;
import ode.vertex.conf.helper.InputStep;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;

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

        final GraphBuilder builder = LayerUtil.initGraphBuilder(Nd4j.getRandom().nextLong(), nrofSamples);
        builder.addInputs("spiral", "time");

        String next = enc.add(builder, "spiral");
        final String qz0_meanAndLogvar = next;

        // Add sampling of a gaussian with the encoded mean and log(var)
        builder.addVertex("z0", new SampleGaussianVertex(Nd4j.getRandom().nextLong()), next);

        next = ode.add(builder, "z0", "time"); // Position of "time" is dependent on argument in constructor to InputStep above
        next = dec.add(builder, next);

        // Steps after this is just for ELBO calculation
        String output0 = outReconstruction.add(builder, next);
        String output1 = outKld.add(builder, qz0_meanAndLogvar);

        builder.setOutputs(output0, output1);

        final ComputationGraph graph = new ComputationGraph(builder.build());
        graph.init();

        LayerUtil.initBiases(graph, WeightInit.UNIFORM);
        return graph;
    }

    @Override
    public String name() {
        return "odenet_enc" + encoderNrofHidden + "_lat" + latentNrofHidden + "_dec" + decoderNrofHidden;
    }


    @Override
    public MultiDataSetPreProcessor getPreProcessor(long nrofLatentDims) {
        return new AddKLDLabel(0, 1, nrofLatentDims);
    }
}
