package examples.cifar10;

import ch.qos.logback.classic.Level;
import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParametersDelegate;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIteratorFactory;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import util.listen.training.NanScoreWatcher;
import util.listen.training.PlotActivations;
import util.listen.training.PlotScore;
import util.listen.training.ZeroGrad;
import util.plot.NoPlot;
import util.plot.Plot;
import util.plot.RealTimePlot;
import util.random.SeededRandomFactory;

import java.io.File;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

/**
 * Example of supervised learning on CIFAR10.
 *
 * @author Christian Skarby
 */
class Main {

    private static final Logger log = LoggerFactory.getLogger(Main.class);

    static final String CHECKPOINT_NAME = "last_checkpoint.zip";
    private static final String BEST_EVAL_NAME = "best_epoch_";

    @Parameter(names = {"-help", "-h"}, description = "Prints help message")
    private boolean help = false;

    @Parameter(names = "-nrofEpochs", description = "Number of epochs to train over")
    private int nrofEpochs = 50;

    @Parameter(names = "-plotTime", description = "Set to plot how time steps evolve over training iterations")
    private boolean plotTime = false;

    @Parameter(names = "-plotScore", description = "Set to plot score for each training iteration")
    private boolean plotScore = false;

    @Parameter(names = "-plotEvalAccuracy", description = "Set to plot accuracy after each epoch")
    private boolean plotEvalAccuracy = false;

    @Parameter(names = "-newModel", description = "Set to true to overwrite any existing model")
    private boolean newModel = false;

    @Parameter(names = "-saveDir", description = "Directory to save models in")
    private String saveDir = "savedmodels";

    @ParametersDelegate
    DataSetIteratorFactory trainFactory = new Cifar10TrainDataProvider();

    @ParametersDelegate
    DataSetIteratorFactory evalFactory = new Cifar10TestDataProvider();

    private ComputationGraph model;
    private ModelFactory modelFactory;

    public static void main(String[] args) throws IOException {
        ch.qos.logback.classic.Logger root = (ch.qos.logback.classic.Logger) LoggerFactory.getLogger(Logger.ROOT_LOGGER_NAME);
        root.setLevel(Level.INFO);

        SeededRandomFactory.setNd4jSeed(0);

        final Main main = parseArgs(args);

        if (!main.help) {
            main.addListeners();
            main.run();
        }
    }

    static Main parseArgs(String... args) {

        final Main main = new Main();
        final Map<String, ModelFactory> modelCommands = new HashMap<>();
        modelCommands.put("odenet", new OdeNetModel());
        modelCommands.put("resnet", new InceptionResNetV1());

        JCommander.Builder parbuilder = JCommander.newBuilder()
                .addObject(main);

        for (Map.Entry<String, ModelFactory> command : modelCommands.entrySet()) {
            parbuilder.addCommand(command.getKey(), command.getValue());
        }

        JCommander jCommander = parbuilder.build();
        jCommander.parse(args);

        if (main.help) {
            jCommander.usage();
            return main;
        }

        final ModelFactory factory = modelCommands.get(jCommander.getParsedCommand());

        main.init(
                main.newModel
                        ? factory
                        : new DeserializingModelFactory(Paths.get(main.saveDir(factory), CHECKPOINT_NAME).toFile(), factory));
        return main;
    }

    void init(ModelFactory factory) {
        this.modelFactory = factory;
        this.model = factory.create();
        long cnt = 0;
        for (GraphVertex vertex : model.getVertices()) {
            log.trace("vertex: " + vertex.getVertexName() + " nrof params: " + vertex.numParams());
            cnt += vertex.numParams();
        }
        log.info("Nrof parameters in model: " + cnt);
    }

    void addListeners() {
        final File savedir = new File(saveDir());
        log.info("Models will be saved in: " + savedir.getAbsolutePath());
        savedir.mkdirs();
        model.addListeners(
                new ZeroGrad(),
                new PerformanceListener(10, true),
                new NanScoreWatcher(() -> {
                    throw new IllegalStateException("NaN score!");
                }));

        if (plotTime) {
            setupTimePlot(savedir);
        }

        if (plotScore) {
            final Plot<Integer, Double> scorePlot = new RealTimePlot<>("Training score", savedir.getAbsolutePath());
            model.addListeners(new PlotScore(scorePlot),
                    new PlotScore(scorePlot, 0.05),
                    new EpochHook(1, new Runnable() {
                        @Override
                        public void run() {
                            try {
                                scorePlot.storePlotData();
                            } catch (IOException e) {
                                throw new UncheckedIOException(e);
                            }
                        }
                    }));
        }
    }

    private void setupTimePlot(File savedir) {
        for(String vertexName: model.getConfiguration().getTopologicalOrderStr()) {

            if(vertexName.startsWith("time")) {

                final Plot<Integer, Double> timePlot = new RealTimePlot<>("Time steps to integrate over " + vertexName, savedir.getAbsolutePath());
                model.addListeners(
                        new PlotActivations(timePlot, vertexName, new String[]{"t0", "t1"}),
                        new EpochHook(1, new Runnable() {
                            @Override
                            public void run() {
                                try {
                                    timePlot.storePlotData();
                                } catch (IOException e) {
                                    throw new UncheckedIOException(e);
                                }
                            }
                        }));
            }
        }
    }


    void run() throws IOException {
        final MultiDataSetIterator trainIter = modelFactory.wrapIter(trainFactory.create());
        final MultiDataSetIterator evalIter = modelFactory.wrapIter(evalFactory.create());

        Nd4j.getMemoryManager().setAutoGcWindow(5000);
        double bestAccuracy = 0;

        final Plot<Integer, Double> evalPlot = plotEvalAccuracy ? new RealTimePlot<>("Eval Accuracy", saveDir()) : new NoPlot<>();
        for (int epoch = model.getEpochCount(); epoch < nrofEpochs; epoch++) {
            log.info("Begin epoch " + epoch);
            model.fit(trainIter);
            log.info("Begin validation in epoch " + epoch);
            final Evaluation evaluation = model.evaluate(evalIter);
            log.info(evaluation.stats() + "\nBest accuracy so far: " + bestAccuracy);

            evalPlot.plotData("Accuracy", epoch, evaluation.accuracy());
            if (evaluation.accuracy() > bestAccuracy) {
                bestAccuracy = evaluation.accuracy();
                model.save(Paths.get(saveDir(), BEST_EVAL_NAME + epoch + ".zip").toFile());
                evalPlot.plotData("Best accuracy", epoch, evaluation.accuracy());
            }
            evalPlot.storePlotData();
            model.save(Paths.get(saveDir(), CHECKPOINT_NAME).toFile());
        }
    }

    String saveDir() {
        return saveDir(modelFactory);
    }

    String saveDir(ModelFactory modelFactory) {
        return Paths.get(saveDir, "CIFAR10", modelFactory.name()).toString();
    }

}
