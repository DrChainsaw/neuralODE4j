package examples.anode;

import com.beust.jcommander.Parameter;
import ode.solve.api.FirstOrderSolverConf;
import ode.solve.conf.DormandPrince54Solver;
import ode.solve.conf.FirstOrderIntegratorConf;
import ode.solve.conf.SolverConfig;
import ode.vertex.conf.helper.FixedStep;
import ode.vertex.conf.helper.OdeHelper;
import org.apache.commons.math3.ode.nonstiff.ClassicalRungeKuttaIntegrator;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.graph.PreprocessorVertex;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.factory.Nd4j;
import util.preproc.ConcatZeros;

/**
 * Create the ODE model described in appendix E, E1.1 and E2.1 in https://arxiv.org/pdf/1904.01681.pdf
 *
 * @author Christian Skarby
 */
class OdeNetModelFactory implements ModelFactory {

    @Parameter(names = "-nrofHidden", description = "Number of hidden units")
    private long nrofHidden = 32;

    @Parameter(names = "-nrofAugmentDims", description = "Number of extra dimensions for augmentation")
    private long nrofAugmentDims = 0;

    @Parameter(names = "-useRk4", description = "Set to use Runge-Kutta 4 instead of Dormand-Prince 5(4)")
    private boolean useRk4 = false;

    @Override
    public Model create(long nrofInputDims) {

        String next = "input";
        final ComputationGraphConfiguration.GraphBuilder builder = LayerUtil.initGraphBuilder(nrofInputDims)
                .addInputs(next);

        if (nrofAugmentDims > 0) {
            builder.addVertex("aug", new PreprocessorVertex(new ConcatZeros(nrofAugmentDims)), next);
            next = "aug";
        }

        final Block odeFunc = new MlpBlock(nrofHidden, nrofInputDims + nrofAugmentDims);

        final FirstOrderSolverConf odeSolverConf = createSolver();

        final OdeHelper odeHelper = new FixedStep(odeSolverConf, Nd4j.arange(2).castTo(Nd4j.defaultFloatingPointType()), true);

        next = new OdeBlockWithTime(
                builder.getGlobalConfiguration(),
                odeFunc,
                odeHelper)
                .add(new GraphBuilderWrapper.Wrap(builder), next);

        final String output = new LossBlock().add(new GraphBuilderWrapper.Wrap(builder), next);
        builder.setOutputs(output);

        final ComputationGraph graph = new ComputationGraph(builder.build());
        graph.init();

        return new OdeNetModel(graph, odeSolverConf,
                (nrofAugmentDims > 0 ? "odenet_aug_" + nrofAugmentDims : "odenet") +
                        (useRk4 ? "_rk4" : "_dopri54"));
    }

    private FirstOrderSolverConf createSolver() {
        // According to appendix E of paper Runge-Kutta 4 is used, but it also says an error tolerance of 1e-3 is used.
        // This does not make sense to me as rk4 is a non-adaptive solver so it does not use tolerances.
        // Rk4 would also not be able to produce the result in figure 7 due to fixed step size.

        if (useRk4) {
            FirstOrderIntegratorConf.addIntegrator(ClassicalRungeKuttaIntegrator.class.getName(), sc -> new ClassicalRungeKuttaIntegrator(0.2));
            return new FirstOrderIntegratorConf(ClassicalRungeKuttaIntegrator.class.getName(),
                    new SolverConfig(1e-3, 1e-3, 1e-10, 1e2));
        } else {
            return new DormandPrince54Solver();
        }
    }
}
