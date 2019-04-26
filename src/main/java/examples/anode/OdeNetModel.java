package examples.anode;

import com.beust.jcommander.Parameter;
import ode.solve.conf.DormandPrince54Solver;
import ode.vertex.conf.helper.FixedStep;
import ode.vertex.conf.helper.OdeHelper;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Create the ODE model described in section E1.1 and E2.1 in https://arxiv.org/pdf/1904.01681.pdf
 *
 * @author Christian Skarby
 */
class OdeNetModel implements ModelFactory {

    @Parameter(names = "-nrofHidden", description = "Number of hidden units")
    private long nrofHidden = 32;

    @Override
    public ComputationGraph create(long nrofInputDims) {

        String next = "input";
        final ComputationGraphConfiguration.GraphBuilder builder = LayerUtil.initGraphBuilder(nrofInputDims)
                .addInputs(next);

        final Block odeFunc = new MlpBlock(nrofHidden, nrofInputDims);
        final OdeHelper odeHelper = new FixedStep(new DormandPrince54Solver(), Nd4j.arange(2), true);

        next = new OdeBlockWithTime(
                builder.getGlobalConfiguration(),
                odeFunc,
                odeHelper)
                .add(new GraphBuilderWrapper.Wrap(builder), next);

        final String output = new LossBlock().add(new GraphBuilderWrapper.Wrap(builder), next);
        builder.setOutputs(output);

        final ComputationGraph graph = new ComputationGraph(builder.build());
        graph.init();

        return graph;
    }
}
