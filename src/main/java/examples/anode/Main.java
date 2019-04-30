package examples.anode;

import ch.qos.logback.classic.Level;
import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParametersDelegate;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIteratorFactory;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import util.listen.training.NanScoreWatcher;
import util.listen.training.PlotScore;
import util.listen.training.ZeroGrad;
import util.plot.Plot;
import util.plot.RealTimePlot;

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

    @Parameter(names = "-plotScore", description = "Set to plot score for each training iteration")
    private boolean plotScore = false;

    @ParametersDelegate
    private DataSetIteratorFactory dataSetIteratorFactory = new AnodeToyDataSetFactory();

    Plot.Factory plotFactory = new RealTimePlot.Factory("");

    Model model;
    private DataSetIterator dataSetIterator;

    public static void main(String[] args) {
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
        modelCommands.put("odenet", new OdeNetModelFactory());

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
        this.dataSetIterator = dataSetIteratorFactory.create();
        this.model = factory.create(dataSetIterator.inputColumns());
        long cnt = 0;
        for (GraphVertex vertex : model.graph().getVertices()) {
            log.trace("vertex: " + vertex.getVertexName() + " nrof params: " + vertex.numParams());
            cnt += vertex.numParams();
        }
        log.info("Nrof parameters in model: " + cnt);
    }

    void addListeners() {
        model.graph().addListeners(
                new ZeroGrad(),
                new PerformanceListener(1, true),
                new NanScoreWatcher(() -> {
                    throw new IllegalStateException("NaN score!");
                }));

        if (plotScore) {
            final Plot<Integer, Double> scorePlot = new RealTimePlot<>("Training score", "");
            model.graph().addListeners(new PlotScore(scorePlot),
                    new PlotScore(scorePlot, 0.05));
        }
    }

    void run() {
        Nd4j.getMemoryManager().setAutoGcWindow(5000);
        final Plot<Double, Double> featurePlot = plotFactory.newPlot("Features evolution");

        for(int epoch = model.graph().getEpochCount(); epoch < nrofEpochs; epoch++) {
            model.graph().fit(dataSetIterator);

            log.info("Epoch " + epoch + " complete! Visualizing features");
            dataSetIterator.reset();
            model.plotFlow(dataSetIterator.next(-1), featurePlot);
            dataSetIterator.reset();
        }
    }

}
