package examples.spiral;

import ch.qos.logback.classic.Level;
import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import examples.spiral.listener.IterationHook;
import examples.spiral.listener.PlotDecodedOutput;
import examples.spiral.listener.SpiralPlot;
import org.apache.commons.io.filefilter.WildcardFileFilter;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import util.listen.training.NanScoreWatcher;
import util.listen.training.PlotActivations;
import util.listen.training.ZeroGrad;
import util.plot.NoPlot;
import util.plot.Plot;
import util.plot.RealTimePlot;
import util.random.SeededRandomFactory;

import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * Main class for spiral example. Reimplementation of https://github.com/rtqichen/torchdiffeq/blob/master/examples/latent_ode.py
 *
 * @author Christian Skarby
 */
class Main {

    private static final Logger log = LoggerFactory.getLogger(Main.class);

    static final String CHECKPOINT_NAME = "last_checkpoint.zip";

    @Parameter(names = {"-help", "-h"}, description = "Prints help message")
    private boolean help = false;

    @Parameter(names = "-trainBatchSize", description = "Batch size to use for training")
    private int trainBatchSize = 1000;

    @Parameter(names = "-nrofTimeStepsForTraining", description = "Number of time steps per spiral when training")
    private int nrofTimeStepsForTraining = 100;

    @Parameter(names = "-nrofTrainIters", description = "Number of iterations for training")
    private int nrofTrainIters = 2000;

    @Parameter(names = "-noiseSigma", description = "How much noise to add to generated spirals")
    private double noiseSigma = 0.3;

    @Parameter(names = "-nrofLatentDims", description = "Number of latent dimensions to use")
    private long nrofLatentDims = 4;

    @Parameter(names = "-saveDir", description = "Directory to save models in")
    private String saveDir = "savedmodels";

    @Parameter(names = "-newModel", description = "Load latest checkpoint (if available) if set to false. If true or if " +
            "no checkpoint exists, a new model will be created")
    private boolean newModel = false;

    @Parameter(names = "-saveEveryNIterations", description = "Sets how often to save a checkpoint")
    private int saveEveryNIterations = 100;

    @Parameter(names = "-noPlot", description = "Set to suppress plotting of training progress and results")
    private boolean noPlot = false;

    private TimeVae model;
    private String modelName;
    private SpiralIterator iterator;
    private Plot.Factory plotBackend;

    public static void main(String[] args) throws IOException {
        ch.qos.logback.classic.Logger root = (ch.qos.logback.classic.Logger) LoggerFactory.getLogger(Logger.ROOT_LOGGER_NAME);
        root.setLevel(Level.INFO);

        SeededRandomFactory.setNd4jSeed(0);

        final Main main = new Main();
        final ModelFactory factory = parseArgs(main, args);

        if (!main.help) {
            createModel(main, factory);
            main.addListeners();
            main.run();
        }
    }

    static ModelFactory parseArgs(Main main, String... args) {

        final Map<String, ModelFactory> modelCommands = new HashMap<>();
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
        }

        return modelCommands.get(jCommander.getParsedCommand());
    }

    @NotNull
    static Main createModel(Main main, ModelFactory factory) throws IOException {
        final String saveDir = main.saveDir(factory.name());

        final ModelFactory factoryToUse;
        if (!main.newModel) {
            factoryToUse = new DeserializingModelFactory(Paths.get(saveDir, CHECKPOINT_NAME).toFile(), factory);
        } else {

            // Else, delete all saved plots and initialize a new model
            final File[] plotFiles = new File(saveDir).listFiles((FilenameFilter) new WildcardFileFilter("*.plt"));
            if (plotFiles != null) {
                for (File plotFile : plotFiles) {
                    Files.delete(Paths.get(plotFile.getAbsolutePath()));
                }
            }
            factoryToUse = factory;
        }

        main.init(
                factoryToUse.createNew(main.nrofTimeStepsForTraining, main.noiseSigma, main.nrofLatentDims),
                factoryToUse.name(),
                factoryToUse.getPreProcessor(main.nrofLatentDims));
        return main;
    }

    String saveDir() {
        return saveDir(modelName);
    }

    String saveDir(String modelName) {
        return Paths.get(saveDir, "spiral", modelName).toString();
    }

    private void init(TimeVae model, String modelName, MultiDataSetPreProcessor preProcessor) {
        this.model = model;
        this.modelName = modelName;

        plotBackend = noPlot ? new NoPlot.Factory() : new RealTimePlot.Factory(saveDir());

        final SpiralFactory spiralFactory = new SpiralFactory(0, 0.3, 0, 6 * Math.PI, 500);
        this.iterator = new SpiralIterator(
                new SpiralIterator.Generator(spiralFactory, noiseSigma, nrofTimeStepsForTraining, new Random(Nd4j.getRandom().nextLong())),
                trainBatchSize);
        iterator.setPreProcessor(preProcessor);
    }

    void addListeners() {
        final File savedir = new File(saveDir());
        log.info("Models will be saved in: " + savedir.getAbsolutePath());
        savedir.mkdirs();

        setupOutputPlotting();
        setupMeanAndLogVarPlotting();

        model.trainingModel().addListeners(
                new ZeroGrad(),
                new PerformanceListener(1, true),
                new NanScoreWatcher(() -> {
                    throw new IllegalStateException("NaN score!");
                }));
    }

    private void setupOutputPlotting() {
        final SpiralPlot outputPlot = new SpiralPlot(plotBackend.newPlot("Training Output"));
        for (int batchNrToPlot = 0; batchNrToPlot < Math.min(trainBatchSize, 4); batchNrToPlot++) {
            outputPlot.plot("True output " + batchNrToPlot, iterator.next().getLabels(0), batchNrToPlot);
            model.trainingModel().addListeners(new PlotDecodedOutput(outputPlot, model.outputName(), batchNrToPlot));
        }
    }

    private void setupMeanAndLogVarPlotting() {
        final Plot<Integer, Double> meanAndLogVarPlot = plotBackend.newPlot("Mean and log(var) of z0");
        model.trainingModel().addListeners(new PlotActivations(meanAndLogVarPlot, model.qzMeanAndLogVarName(), new String[]{"qz0Mean", "qz0Log(Var)"}),
                new IterationHook(saveEveryNIterations, () -> {
                    try {
                        meanAndLogVarPlot.storePlotData();
                    } catch (IOException e) {
                        log.error("Could not save plot data! Exception:" + e.getMessage());
                    }
                }));
    }


    void run() throws IOException {
        final ComputationGraph trainingModel = model.trainingModel();

        final Plot<Double, Double> samplePlot = plotBackend.newPlot("Reconstruction");

        for (int i = trainingModel.getIterationCount(); i < nrofTrainIters; i++) {
            trainingModel.fit(iterator.next());


            if (i > 0 && i % 100 == 0) {
                drawSample(0, samplePlot);
                samplePlot.savePicture("_iter" + trainingModel.getIterationCount());
            }

            if (i > 0 && i % saveEveryNIterations == 0) {
                trainingModel.save(Paths.get(saveDir(), CHECKPOINT_NAME).toFile());
            }
        }

        for (int i = 0; i < Math.min(trainBatchSize, 8); i++) {
            final Plot<Double, Double> plot = plotBackend.newPlot("Reconstruction " + i);
            drawSample(i, plot);
            plot.savePicture("");
        }
    }

    private void drawSample(final int toSample, Plot<Double, Double> reconstructionPlot) {
        log.info("Sampling model...");

        final SpiralIterator.SpiralSet spiralSet = iterator.getCurrent();
        final MultiDataSet mds = spiralSet.getMds();
        final INDArray sample = mds.getFeatures(0).tensorAlongDimension(toSample, 1, 2).reshape(1, 2, nrofTimeStepsForTraining);

        final INDArray z0 = model.encode(sample);

        final INDArray tsPos = Nd4j.linspace(0, 2 * Math.PI, 2000);
        final INDArray tsNeg = Nd4j.linspace(0, -Math.PI, 2000);

        final INDArray zsPos = model.timeDependency(z0, tsPos);
        final INDArray zsNeg = model.timeDependency(z0, tsNeg);

        final INDArray xsPos = model.decode(zsPos);
        final INDArray xsNeg = model.decode(zsNeg);

        final SpiralPlot spiralPlot = new SpiralPlot(reconstructionPlot);

        reconstructionPlot.clearData("True trajectory");
        reconstructionPlot.clearData("Sampled data");
        reconstructionPlot.clearData("Learned trajectory (t > 0)");
        reconstructionPlot.clearData("Learned trajectory (t < 0)");

        spiralSet.getSpirals().get(toSample).plotBase(reconstructionPlot, "True trajectory");
        spiralPlot.plot("Sampled data", sample, 0); // Always dim 0 as shape is [1, 2, nrofTimeSteps]
        spiralPlot.plot("Learned trajectory (t > 0)", xsPos, 0); // Always dim 0 as shape is [1, 2, 2000]
        spiralPlot.plot("Learned trajectory (t < 0)", xsNeg, 0); // Always dim 0 as shape is [1, 2, 2000]
    }
}
