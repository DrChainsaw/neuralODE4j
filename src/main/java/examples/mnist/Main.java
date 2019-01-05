package examples.mnist;

import ch.qos.logback.classic.Level;
import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.optimize.listeners.CheckpointListener;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import util.listen.training.NanScoreWatcher;
import util.listen.training.ZeroGrad;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * Main class for MNIST example. Reimplementation of https://github.com/rtqichen/torchdiffeq/blob/master/examples/odenet_mnist.py
 *
 * @author Christian Skarby
 */
class Main {

    private static final Logger log = LoggerFactory.getLogger(Main.class);

    @Parameter(names = "-trainBatchSize", description = "Batch size to use for training")
    private int trainBatchSize = 128;

    @Parameter(names = "-evalBatchSize", description = "Batch size to use for validation")
    private int evalBatchSize = 1000;

    @Parameter(names = "-nrofEpochs", description = "Number of epochs to train over")
    private int nrofEpochs = 200;

    @Parameter(names = "-nrofTrainExamples", description = "Number of examples to use for training")
    private int nrofTrainExamples = MnistDataFetcher.NUM_EXAMPLES;

    @Parameter(names = "-nrofTestExamples", description = "Number of examples to use for validation")
    private int nrofTestExamples = MnistDataFetcher.NUM_EXAMPLES_TEST;

    private ComputationGraph model;
    private String modelName;

    public static void main(String[] args) throws IOException {
        ch.qos.logback.classic.Logger root = (ch.qos.logback.classic.Logger) LoggerFactory.getLogger(Logger.ROOT_LOGGER_NAME);
        root.setLevel(Level.INFO);

        final Main main = parseArgs(args);

        main.addListeners();
        main.run();
    }

    private static Main parseArgs(String[] args) {

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

        final ModelFactory factory = modelCommands.get(jCommander.getParsedCommand());

        main.init(factory.create(), jCommander.getParsedCommand());
        return main;
    }

    private void init(ComputationGraph model, String modelName) {
        this.model = model;
        this.modelName = modelName;
        long cnt = 0;
        for(GraphVertex vertex : model.getVertices()) {
            log.trace("vertex: " + vertex.getVertexName() + " nrof params: " + vertex.numParams());
            cnt += vertex.numParams();
        }
        log.info("Nrof parameters in model: " + cnt);
    }

    private void addListeners() {
        model.addListeners(
                new ZeroGrad(),
                new PerformanceListener(20, true),
                new CheckpointListener.Builder(new File(modelName).getAbsolutePath())
                        .keepLast(1)
                        .deleteExisting(true)
                        .saveEveryEpoch()
                        .build(),
                new NanScoreWatcher(() -> {
                    throw new IllegalStateException("NaN score!");
                }));
    }

    private void run() throws IOException {
        final DataSetIterator trainIter = new MnistDataSetIterator(trainBatchSize, nrofTrainExamples, false, true, true, 1234);
        final DataSetIterator evalIter = new MnistDataSetIterator(evalBatchSize, nrofTestExamples, false, false, false, 1234);
        Nd4j.getMemoryManager().setAutoGcWindow(5000);

        double bestAccuracy = 0;
        for (int epoch = model.getEpochCount(); epoch < nrofEpochs; epoch++) {
            log.info("Begin epoch " + epoch);
            model.fit(trainIter);
            log.info("Begin validation in epoch " + epoch);
            final Evaluation evaluation = model.evaluate(evalIter);
            log.info(evaluation.stats() + " best accuracy so far: " + bestAccuracy);

            if(evaluation.accuracy() > bestAccuracy) {
                bestAccuracy = evaluation.accuracy();
                model.save(new File("best_" + modelName + "_epoch_" + epoch));
            }
        }
    }
}
