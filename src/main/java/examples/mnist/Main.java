package examples.mnist;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.listeners.CheckpointListener;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

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

    private ComputationGraph model;

    public static void main(String[] args) throws IOException {
        final ResNetReferenceModel referenceModel = new ResNetReferenceModel();
        final OdeNetModel odeModel = new OdeNetModel();
        final Main main = new Main();
        JCommander.newBuilder()
                .addObject(new Object[]{
                        referenceModel,
                        main})
                .addCommand("resnet", referenceModel)
                .addCommand("odenet", odeModel)
                .build()
                .parse(args);

        main.init(odeModel.create());
        main.addListeners();
        main.run();
    }

    void init(ComputationGraph model) {
        this.model = model;
        log.info("Nrof parameters in model: " + model.numParams());
    }

    void addListeners() {
        model.addListeners(
                new PerformanceListener(1, true),
                new CheckpointListener.Builder("")
                        .keepLast(1)
                        .deleteExisting(true)
                        .saveEveryNIterations(1000)
                        .build());
    }

    void run() throws IOException {
        final DataSetIterator trainIter = new MnistDataSetIterator(trainBatchSize, true, 666);
        final DataSetIterator evalIter = new MnistDataSetIterator(evalBatchSize, false, 666);
        model.getLayers();
        for (int epoch = 0; epoch < nrofEpochs; epoch++) {
            log.info("Begin epoch " + epoch);
            model.fit(trainIter);
            log.info("Begin validation in epoch " + epoch);
            final Evaluation evaluation = model.evaluate(evalIter);
            log.info(evaluation.stats());
        }

    }


}
