package examples.cifar10;

import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import examples.cifar10.GraphBuilderWrapper.Wrap;
import ode.solve.api.FirstOrderSolverConf;
import ode.solve.conf.DormandPrince54Solver;
import ode.solve.conf.SolverConfig;
import ode.vertex.conf.helper.InputStep;
import ode.vertex.conf.helper.OdeHelper;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import util.listen.step.Mask;
import util.listen.step.StepCounter;

import static examples.cifar10.LayerUtil.conv1x1;

/**
 * OdeNet reference model for. Reimplementation of https://github.com/rtqichen/torchdiffeq/blob/master/examples/odenet_mnist.py
 *
 * @author Christian Skarby
 */
@Parameters(commandDescription = "Configuration for image classification using an ODE block")
public class OdeNetModel implements ModelFactory {

    private static final Logger log = LoggerFactory.getLogger(OdeNetModel.class);

    @Parameter(names = "-nrofPreReductions", description = "Number of type A reduction blocks to apply before the first ODE block.")
    private int nrofPreReductions = 0;

    @Parameter(names = "-useABlock", description = "Use an ODE A block if true.")
    private boolean useABlock = false;

    @Parameter(names = "-useBBlock", description = "Use an ODE B block if true.")
    private boolean useBBlock = false;

    @Parameter(names = "-useCBlock", description = "Use an ODE C block if true.")
    private boolean useCBlock = false;

    @Parameter(names = "-trainTime", description = "Set if time steps for solver shall be trained")
    private boolean trainTime = false;

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

        String next = "input";
        final String time = "time";
        final GraphBuilder builder = LayerUtil.initGraphBuilder(Nd4j.getRandom().nextLong(), next, time);

        OdeHelper odeConf = new InputStep(solver, 1, false, false);

        if (trainTime) {
            odeConf = new InputStep(solver, 1, false, true);
        }

        for(int i = 0; i < nrofPreReductions; i++) {
            next = new InceptionReductionABlock("preReductionA" + i).add(new Wrap(builder), next);
        }

        builder.addLayer("resize", conv1x1(256), next);
        next = "resize";

        if(useABlock) {
            final String timeName = addTimeIfTimeTrain("A", builder, time);
            next = new OdeBlockTime(builder.getGlobalConfiguration(), odeConf, new InceptionResNetABlock("", 32), "odeVertexA")
                    .add(new Wrap(builder), next, timeName);
        }

        if(useBBlock || useCBlock) {
            next = new InceptionReductionABlock("reductionA").add(new Wrap(builder), next);
        }

        if(useBBlock) {
            final String timeName = addTimeIfTimeTrain("B", builder, time);
            next = new OdeBlockTime(builder.getGlobalConfiguration(), odeConf, new InceptionResNetBBlock("", 128), "odeVertexB")
                    .add(new Wrap(builder), next, timeName);
        }

        if(useCBlock) {
            final String timeName = addTimeIfTimeTrain("C", builder, time);
            next = new InceptionReductionBBlock("reductionB").add(new Wrap(builder), next);
            next = new OdeBlockTime(builder.getGlobalConfiguration(), odeConf, new InceptionResNetCBlock("", 128), "odeVertexC")
                    .add(new Wrap(builder), next, timeName);
        }

        next = new Output().add(new Wrap(builder), next);
        builder.setOutputs(next);

        final ComputationGraph graph = new ComputationGraph(builder.build());
        graph.init();
        return graph;
    }

    private String addTimeIfTimeTrain(String blockSuffix, GraphBuilder builder, String timeName) {
        if (trainTime) {
            builder.addLayer(timeName + blockSuffix, new DenseLayer.Builder()
                    .nOut(2)
                    .weightInit(WeightInit.IDENTITY)
                    .biasInit(0)
                    .activation(new ActivationIdentity())
                    .build(), timeName);
            return timeName + blockSuffix;
        }
        return timeName;
    }

    @Override
    public String name() {
        return "odenet"
                + (nrofPreReductions > 0 ? "_nrofPreReductions" + nrofPreReductions : "")
                + (useABlock ? "_A" : "")
                + (useBBlock ? "_B" : "")
                + (useCBlock ? "_C" : "")
                + (trainTime ? "_trainTime" : "");
    }

    @Override
    public MultiDataSetIterator wrapIter(DataSetIterator iter) {
        return
                new AddTimeMultiDataSetIter(
                        iter
                        , Nd4j.arange(2).castTo(Nd4j.defaultFloatingPointType())); // Start with time steps 0 to 1
    }
}

