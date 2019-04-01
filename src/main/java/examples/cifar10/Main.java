package examples.cifar10;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import org.datavec.image.loader.CifarLoader;
import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.optimize.listeners.CheckpointListener;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.CompositeDataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import util.listen.training.NanScoreWatcher;
import util.listen.training.PlotActivations;
import util.listen.training.PlotScore;
import util.listen.training.ZeroGrad;
import util.plot.Plot;
import util.plot.RealTimePlot;
import util.preproc.ShiftDim;
import util.random.SeededRandomFactory;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * Example of supervised learning on CIFAR10.
 *
 * @author Christian Skarby
 */
class Main {

    private static final Logger log = LoggerFactory.getLogger(Main.class);

    @Parameter(names = {"-help", "-h"}, description = "Prints help message")
    private boolean help = false;

    @Parameter(names = "-trainBatchSize", description = "Batch size to use for training")
    private int trainBatchSize = 32;

    @Parameter(names = "-evalBatchSize", description = "Batch size to use for validation")
    private int evalBatchSize = 32;

    @Parameter(names = "-nrofEpochs", description = "Number of epochs to train over")
    private int nrofEpochs = 50;

    @Parameter(names = "-nrofTrainExamples", description = "Number of examples to use for training")
    private int nrofTrainExamples = CifarLoader.NUM_TRAIN_IMAGES;

    @Parameter(names = "-nrofTestExamples", description = "Number of examples to use for validation")
    private int nrofTestExamples = CifarLoader.NUM_TEST_IMAGES;

    @Parameter(names = "-dataAug", description = "Use data augmentation for training if set to true", arity = 1)
    private boolean useDataAugmentation = true;

    @Parameter(names = "-plotTime", description = "Set to true to plot how time steps evolve over training iterations")
    private boolean plotTime = false;

    @Parameter(names = "-plotScore", description = "Set to true to plot how time steps evolve over training iterations")
    private boolean plotScore = false;

    private ComputationGraph model;
    private String modelName;
    private ModelFactory modelFactory;

    public static void main(String[] args) throws IOException {
        ch.qos.logback.classic.Logger root = (ch.qos.logback.classic.Logger) LoggerFactory.getLogger(Logger.ROOT_LOGGER_NAME);
        //root.setLevel(Level.INFO);

        SeededRandomFactory.setNd4jSeed(0);

        final Main main = parseArgs(args);

        if (!main.help) {
            main.addListeners();
            main.run();
        }
    }

    private static Main parseArgs(String[] args) {

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
        }

        final ModelFactory factory = modelCommands.get(jCommander.getParsedCommand());

        main.init(factory);
        return main;
    }

    private void init(ModelFactory factory) {
        this.modelFactory = factory;
        this.model = factory.create();
        this.modelName = factory.name();
        long cnt = 0;
        for (GraphVertex vertex : model.getVertices()) {
            log.trace("vertex: " + vertex.getVertexName() + " nrof params: " + vertex.numParams());
            cnt += vertex.numParams();
        }
        log.info("Nrof parameters in model: " + cnt);
    }

    private void addListeners() {
        final File savedir = saveDir(modelName);
        log.info("Models will be saved in: " + savedir.getAbsolutePath());
        savedir.mkdirs();
        model.addListeners(
                new ZeroGrad(),
                new PerformanceListener(1, true),
                new CheckpointListener.Builder(savedir.getAbsolutePath())
                        .keepLast(1)
                        .deleteExisting(true)
                        .saveEveryEpoch()
                        .build(),
                new NanScoreWatcher(() -> {
                    throw new IllegalStateException("NaN score!");
                }));

        if (plotTime) {
            final Plot<Integer, Double> timePlot = new RealTimePlot<>("Time steps to integrate over", savedir.getAbsolutePath());
            model.addListeners(new PlotActivations(timePlot, "timeTrain", new String[]{"t0", "t1"}));
        }

        if (plotScore) {
            final Plot<Integer, Double> scorePlot = new RealTimePlot<>("Training score", savedir.getAbsolutePath());
            model.addListeners(new PlotScore(scorePlot));
        }
    }

    private void run() throws IOException {
        final MultiDataSetIterator trainIter = createDataSetIter(true);
        final MultiDataSetIterator evalIter = createDataSetIter(false);

        Nd4j.getMemoryManager().setAutoGcWindow(5000);
        double bestAccuracy = 0;
        for (int epoch = model.getEpochCount(); epoch < nrofEpochs; epoch++) {
            log.info("Begin epoch " + epoch);
            model.fit(trainIter);
            log.info("Begin validation in epoch " + epoch);
            final Evaluation evaluation = model.evaluate(evalIter);
            log.info(evaluation.stats() + "\nBest accuracy so far: " + bestAccuracy);

            if (evaluation.accuracy() > bestAccuracy) {
                bestAccuracy = evaluation.accuracy();
                model.save(saveDir(modelName + File.separator +  "best_epoch_" + epoch + ".zip"));
            }
        }
    }

    private static File saveDir(String modelName) {
        return new File("savedmodels" + File.separator + "CIFAR10" + File.separator + modelName);
    }

    private MultiDataSetIterator createDataSetIter(boolean train) {

        final DataSetIterator iter = new CifarDataSetIterator(
                train ? trainBatchSize : evalBatchSize,
                train ? nrofTrainExamples : nrofTestExamples,
                train);

        if (train && useDataAugmentation) {
            iter.setPreProcessor(new CompositeDataSetPreProcessor(
                    new ShiftDim(2, new Random(666), 4),
                    new ShiftDim(3, new Random(667), 4)
            ));
        }

        return modelFactory.wrapIter(iter);
    }
}
