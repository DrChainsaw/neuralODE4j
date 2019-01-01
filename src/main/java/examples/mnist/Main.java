package examples.mnist;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.listeners.CheckpointListener;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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

    public static void main(String[] args) throws IOException {
        final Main main = new Main();

        final ModelFactory modelFactory = parseArgs(args, main);

        main.init(modelFactory.create());
        main.addListeners();
        main.run();
    }

    private static ModelFactory parseArgs(String[] args, Main main) {
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

        return modelCommands.get(jCommander.getParsedCommand());
    }

    private void init(ComputationGraph model) {
        this.model = model;
        log.info("Nrof parameters in model: " + model.numParams());
    }

    private void addListeners() {
        model.addListeners(
                new PerformanceListener(20, true),
                new CheckpointListener.Builder("")
                        .keepLast(1)
                        .deleteExisting(true)
                        .saveEveryNIterations(1000)
                        .build(),
                new NanScoreWatcher(() -> {
                    throw new IllegalStateException("NaN score!");
                }));
    }

    private void run() throws IOException {
        final DataSetIterator trainIter = new MnistDataSetIterator(trainBatchSize, nrofTrainExamples, false, true, true, 666);
        final DataSetIterator evalIter = new MnistDataSetIterator(evalBatchSize, nrofTestExamples, false, false, true, 666);
        model.getLayers();
        Nd4j.getMemoryManager().setAutoGcWindow(5000);
        for (int epoch = 0; epoch < nrofEpochs; epoch++) {
            log.info("Begin epoch " + epoch);
            model.fit(trainIter);
            log.info("Begin validation in epoch " + epoch);
            final Evaluation evaluation = model.evaluate(evalIter);
            log.info(evaluation.stats());
        }

    }


}
