package examples.spiral;

import examples.spiral.loss.NormElboLoss;
import examples.spiral.loss.NormKLDLoss;
import examples.spiral.loss.NormLogLikelihoodLoss;
import examples.spiral.loss.PredMeanLogvar2D;
import examples.spiral.vertex.conf.SampleGaussianVertex;
import ode.solve.conf.DormandPrince54Solver;
import ode.solve.conf.SolverConfig;
import ode.vertex.conf.OdeVertex;
import ode.vertex.conf.helper.InputStep;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.graph.PreprocessorVertex;
import org.deeplearning4j.nn.conf.graph.SubsetVertex;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LossLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.preprocessors.ReshapePreprocessor;
import org.nd4j.linalg.activations.impl.ActivationELU;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;

/**
 * Model used for spiral generation using neural ODE. Equivalent to model used in
 * https://github.com/rtqichen/torchdiffeq/blob/master/examples/latent_ode.py
 *
 * @author Christian Skarby
 */
class OdeNetModel implements ModelFactory {

    private double noiseSigma;
    private long nrofSamples;
    private long nrofLatentDims;

    @Override
    public ComputationGraph create(long nrofSamples, double noiseSigma, long nrofLatentDims) {
        this.noiseSigma = noiseSigma;
        this.nrofSamples = nrofSamples;
        this.nrofLatentDims = nrofLatentDims;

        Block enc = new RnnEncoderBlock(nrofLatentDims, 25, "spiral");
        Block dec = new DenseDecoderBlock(20, nrofSamples, 2);

        final GraphBuilder builder = LayerUtil.initGraphBuilder(666, nrofSamples);
        builder.addInputs("spiral", "time");

        String next = enc.add("spiral", builder);

        // Add sampling of a gaussian with the encoded mean and log(var)
        final String qz0_mean = "qz0_mean";
        final String qz0_logvar = "qz0_logvar";
        builder
                //First split prev into mean and log(var) since this is how SampleGaussianVertex assumes input is structured
                .addVertex(qz0_mean, new SubsetVertex(0, (int) nrofLatentDims - 1), next)
                .addVertex(qz0_logvar, new SubsetVertex((int) nrofLatentDims, (int) nrofLatentDims * 2 - 1), next)
                .addVertex("z0", new SampleGaussianVertex(666), qz0_mean, qz0_logvar);
                //.addVertex("z0", new ElementWiseVertex(ElementWiseVertex.Op.Add), qz0_mean, qz0_logvar);

        next = addLatentOde("z0", "time", builder);
        next = dec.add(next, builder);
        //final String actualOutput = next;
        // Steps after this is just for ELBO calculation
        addLoss(next, qz0_mean, qz0_logvar, builder);

        final ComputationGraph graph = new ComputationGraph(builder.build());
        graph.init();

        return graph;
    }

    @Override
    public String name() {
        return "odenet";
    }

    private String addLatentOde(String prev, String time, GraphBuilder builder) {
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
                        new DormandPrince54Solver(new SolverConfig(1e-9, 1e-7, 1e-20, 1e2)),
                        1))
                .build(), prev, time);
        return "latentOde";
    }

    private void addLoss(
            String decOut,
            String qz0_mean,
            String qz0_logvar,
            GraphBuilder builder) {
        // First we need to concatenate the following into one single array:
        // 1. The decoded output
        // 2. qz0_mean
        // 3. qz0_logvar
        // This is because the API to the loss function only takes on single input INDArray.
        builder
                // Since 1 above is 3D and the other two are 2D, the first step is to "flatten" 1 into 2D using an ReshapePreprocessor
                .addVertex("flattenDec", new PreprocessorVertex(new ReshapePreprocessor(
                        new long[]{2, nrofSamples},
                        new long[]{2 * nrofSamples})), decOut)
                // Note: Merge order is determined by PredMeanLogvar2D implementation
                .addVertex("merge", new MergeVertex(), "flattenDec", qz0_mean, qz0_logvar)
                .addLayer("loss", new LossLayer.Builder()
                                .activation(new ActivationIdentity())
                                .lossFunction(new NormElboLoss(
                                        new NormLogLikelihoodLoss(noiseSigma),
                                        new NormKLDLoss(),
                                        new PredMeanLogvar2D(nrofLatentDims)))
                                .build()
                        , "merge")
                .setOutputs("loss"); // Not really output, just used for loss calculation
    }

    @Override
    public MultiDataSetPreProcessor getPreProcessor() {
        return new LabelsPreProcessor(new ReshapePreprocessor(new long[]{2, nrofSamples},
                new long[]{nrofLatentDims * nrofSamples / 2}));
    }
}
