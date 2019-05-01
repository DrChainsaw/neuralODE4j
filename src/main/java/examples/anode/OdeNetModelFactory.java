package examples.anode;

import com.beust.jcommander.Parameter;
import ode.solve.api.FirstOrderSolverConf;
import ode.solve.conf.DormandPrince54Solver;
import ode.vertex.conf.helper.FixedStep;
import ode.vertex.conf.helper.OdeHelper;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.graph.PreprocessorVertex;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.factory.Nd4j;
import util.preproc.ConcatZeros;

/**
 * Create the ODE model described in section E1.1 and E2.1 in https://arxiv.org/pdf/1904.01681.pdf
 *
 * @author Christian Skarby
 */
class OdeNetModelFactory implements ModelFactory {

    @Parameter(names = "-nrofHidden", description = "Number of hidden units")
    private long nrofHidden = 32;

    @Parameter(names = "-nrofAugmentDims", description = "Number of extra dimensions for augmentation")
    private long nrofAugmentDims = 0;

    @Override
    public Model create(long nrofInputDims) {

        String next = "input";
        final ComputationGraphConfiguration.GraphBuilder builder = LayerUtil.initGraphBuilder(nrofInputDims)
                .addInputs(next);

        if(nrofAugmentDims > 0) {
            builder.addVertex("aug", new PreprocessorVertex(new ConcatZeros(nrofAugmentDims)), next);
            next = "aug";
        }

        final Block odeFunc = new MlpBlock(nrofHidden, nrofInputDims + nrofAugmentDims);
        final FirstOrderSolverConf odeSolverConf = new DormandPrince54Solver();
        final OdeHelper odeHelper = new FixedStep(odeSolverConf, Nd4j.arange(2), true);

        next = new OdeBlockWithTime(
                builder.getGlobalConfiguration(),
                odeFunc,
                odeHelper)
                .add(new GraphBuilderWrapper.Wrap(builder), next);

        final String output = new LossBlock().add(new GraphBuilderWrapper.Wrap(builder), next);
        builder.setOutputs(output);

        final ComputationGraph graph = new ComputationGraph(builder.build());
        graph.init();

        return new OdeNetModel(graph, odeSolverConf, nrofAugmentDims > 0 ? "odenet_aug_" + nrofAugmentDims : "odenet");
    }
}
