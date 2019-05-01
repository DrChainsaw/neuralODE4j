package examples.anode;

import ch.qos.logback.classic.Level;
import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParametersDelegate;
import examples.cifar10.EpochHook;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import util.listen.training.NanScoreWatcher;
import util.listen.training.PlotScore;
import util.listen.training.ZeroGrad;
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
 * Runs experiments from section 4.1 of augmented neural ODEs: https://arxiv.org/pdf/1904.01681.pdf
 *
 * @author Christian Skarby
 */
class Main {

    private static final Logger log = LoggerFactory.getLogger(Main.class);

    @Parameter(names = {"-help", "-h"}, description = "Prints help message")
    private boolean help = false;

    @Parameter(names = "-nrofEpochs", description = "Number of epochs to train over")
    private int nrofEpochs = 50;

    @Parameter(names = "-saveEveryNEpochs", description = "Save figures every N epochs")
    private int saveEveryNEpochs = 1;

    @Parameter(names = "-saveDir", description = "Directory to save output in")
    private String saveDir = "savedmodels";

    @ParametersDelegate
    private AnodeToyDataSetFactory dataSetIteratorFactory = new AnodeToyDataSetFactory();

    Plot.Factory plotFactory;
    Model model;
    private AnodeToyDataSetFactory.DataSetIters dataSetIterators;

    public static void main(String[] args) {
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
        modelCommands.put("odenet", new OdeNetModelFactory());
        modelCommands.put("resnet", new ResNetModelFactory());

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

        main.init(modelCommands.get(jCommander.getParsedCommand()));
        return main;
    }

    void init(ModelFactory factory) {
        this.dataSetIterators = dataSetIteratorFactory.create();
        this.model = factory.create(dataSetIterators.getTrain().inputColumns());
        long cnt = 0;
        for (GraphVertex vertex : model.graph().getVertices()) {
            log.trace("vertex: " + vertex.getVertexName() + " nrof params: " + vertex.numParams());
            cnt += vertex.numParams();
        }
        log.info("Nrof parameters in model: " + cnt);

        plotFactory = new RealTimePlot.Factory(saveDir());
    }

    void addListeners() {

        final File savedir = new File(saveDir());
        log.info("Plots will be saved in: " + savedir.getAbsolutePath());
        savedir.mkdirs();

        final Plot<Integer, Double> scorePlot = plotFactory.newPlot("Training score");
        final Plot<Double, Double> featurePlot = setupFeaturePlot();

        model.graph().addListeners(
                new ZeroGrad(),
                new PerformanceListener(1, true),
                new NanScoreWatcher(() -> {
                    throw new IllegalStateException("NaN score!");
                }),
                new PlotScore(scorePlot),
                new PlotScore(scorePlot, 0.05),
                new EpochHook(1, () -> {
                    log.info("Epoch " + model.graph().getEpochCount() + " complete! Visualizing features");
                    this.model.plotFeatures(dataSetIterators.getTest().next(), featurePlot);
                    dataSetIterators.getTest().reset();
                }),
                new EpochHook(saveEveryNEpochs, () -> {
                    try {
                        scorePlot.savePicture("");
                        featurePlot.savePicture("_epoch_" + model.graph().getEpochCount());
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                })
        );
    }

    @NotNull
    private Plot<Double, Double> setupFeaturePlot() {
        final Plot<Double, Double> featurePlot = plotFactory.newPlot("Feature space");
        final DataSet ds = dataSetIterators.getTest().next();
        dataSetIterators.getTest().reset();
        PlotState.plotXY(ds.getFeatures(), ds.getLabels(), featurePlot);
        try {
            featurePlot.savePicture("_input");
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        return featurePlot;
    }

    void run() {
        Nd4j.getMemoryManager().setAutoGcWindow(5000);
        for (int epoch = model.graph().getEpochCount(); epoch < nrofEpochs; epoch++) {
            model.graph().fit(dataSetIterators.getTrain());
        }
    }

    String saveDir() {
        return saveDir(model, dataSetIterators);
    }

    String saveDir(Model model, AnodeToyDataSetFactory.DataSetIters dataSetIters) {
        return Paths.get(saveDir, "ANODE", dataSetIters.getName() + "_" + model.name()).toString();
    }
}
