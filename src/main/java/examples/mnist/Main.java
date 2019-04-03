package examples.mnist;

import ch.qos.logback.classic.Level;
import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParametersDelegate;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIteratorFactory;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import util.listen.training.NanScoreWatcher;
import util.listen.training.ZeroGrad;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

/**
 * Main class for MNIST example. Reimplementation of https://github.com/rtqichen/torchdiffeq/blob/master/examples/odenet_mnist.py
 *
 * @author Christian Skarby
 */
class Main {

    private static final Logger log = LoggerFactory.getLogger(Main.class);

    static final String CHECKPOINT_NAME = "last_checkpoint.zip";
    static final String BEST_EVAL_NAME = "best_epoch_";

    @Parameter(names = {"-help", "-h"}, description = "Prints help message")
    private boolean help = false;

    @Parameter(names = "-nrofEpochs", description = "Number of epochs to train over")
    private int nrofEpochs = 160;

    @Parameter(names = "-newModel", description = "Set to true to overwrite any existing model")
    private boolean newModel = false;

    @Parameter(names = "-saveDir", description = "Directory to save models in")
    private String saveDir = "savedmodels";

    @ParametersDelegate
    public DataSetIteratorFactory trainFactory = new MnistTrainDataProvider();

    @ParametersDelegate
    public DataSetIteratorFactory evalFactory = new MnistTestDataProvider();

    private ComputationGraph model;
    private String modelName;

    public static void main(String[] args) throws IOException {
        ch.qos.logback.classic.Logger root = (ch.qos.logback.classic.Logger) LoggerFactory.getLogger(Logger.ROOT_LOGGER_NAME);
        root.setLevel(Level.INFO);

        final Main main = parseArgs(args);

        if (!main.help) {
            main.addListeners();
            main.run();
        }
    }

    static Main parseArgs(String... args) {

        final Main main = new Main();
        final Map<String, ModelFactory> modelCommands = new HashMap<>();
        modelCommands.put("resnet", new ResNetReferenceModel());
        modelCommands.put("odenet", new OdeNetModel());

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

        ModelFactory factory = modelCommands.get(jCommander.getParsedCommand());

        factory = main.newModel
                ? factory
                : new DeserializingModelFactory(Paths.get(main.saveDir(factory.name()), CHECKPOINT_NAME).toFile(), factory);

        main.init(factory.create(), factory.name());
        return main;
    }

    void init(ComputationGraph model, String modelName) {
        this.model = model;
        this.modelName = modelName;
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
                new PerformanceListener(20, true),
                new NanScoreWatcher(() -> {
                    throw new IllegalStateException("NaN score!");
                }));
    }

    void run() throws IOException {
        final DataSetIterator trainIter = trainFactory.create();
        final DataSetIterator evalIter = evalFactory.create();

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
                model.save(Paths.get(saveDir(), BEST_EVAL_NAME + epoch + ".zip").toFile());
            }
            model.save(Paths.get(saveDir(), CHECKPOINT_NAME).toFile());
        }
    }

    String saveDir() {
        return saveDir(modelName);
    }

    String saveDir(String modelName) {
        return Paths.get(saveDir, "MNIST", modelName).toString();
    }
}
