package examples.mnist;

import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParametersDelegate;
import ode.solve.api.FirstOrderSolverConf;
import ode.solve.conf.DormandPrince54Solver;
import ode.solve.conf.SolverConfig;
import ode.vertex.conf.OdeVertex;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import util.listen.step.Mask;
import util.listen.step.StepCounter;

import static examples.mnist.LayerUtil.*;

/**
 * OdeNet reference model for. Reimplementation of https://github.com/rtqichen/torchdiffeq/blob/master/examples/odenet_mnist.py
 *
 * @author Christian Skarby
 */
public class OdeNetModel implements ModelFactory {

    private static final Logger log = LoggerFactory.getLogger(OdeNetModel.class);

    @ParametersDelegate
    private StemSelection stemSelection = new StemSelection();

    @Parameter(names = "-nrofKernels", description = "Number of filter kernels in each convolution layer")
    private int nrofKernels = 64;

    private final GraphBuilder builder;

    public OdeNetModel() {
        this.builder = initGraphBuilder(666);
    }

    @Override
    public ComputationGraph create() {
        return create(
                //new NanWatchSolver(
                new DormandPrince54Solver(new SolverConfig(1e-3, 1e-3, 1e-20, 100)));
    }

    ComputationGraph create(FirstOrderSolverConf solver) {

        log.info("Create model");

        solver.addListeners(
                Mask.backward(new StepCounter(20, "Average nrof forward steps: ")),
                Mask.forward(new StepCounter(20, "Average nrof backward steps: "))
        );

        String next = stemSelection.get(nrofKernels).add(builder.getNetworkInputs().get(0), builder);
        next = addOdeBlock(next, solver);
        new Output(nrofKernels).add(next, builder);
        final ComputationGraph graph = new ComputationGraph(builder.build());
        graph.init();
        return graph;
    }

    @Override
    public String name() {
        return "odenet_" + stemSelection.name();
    }

    private String addOdeBlock(String prev, FirstOrderSolverConf solver) {
        builder
                .addVertex("odeBlock", new OdeVertex.Builder("normFirst",
                        norm(nrofKernels))
                        .addLayer("convFirst",
                                conv3x3Same(nrofKernels), "normFirst")
                        .addLayer("normSecond",
                                norm(nrofKernels), "convFirst")
                        .addLayer("convSecond",
                                conv3x3Same(nrofKernels), "normSecond")
                        .addLayer("normThird",
                                norm(nrofKernels), "convSecond")
                        .odeSolver(solver)
                        .build(), prev);
        return "odeBlock";
    }
}
