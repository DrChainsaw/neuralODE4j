package examples.spiral;

import ch.qos.logback.classic.Level;
import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import examples.spiral.listener.PlotActivations;
import examples.spiral.listener.PlotDecodedOutput;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.optimize.listeners.CheckpointListener;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import util.listen.training.NanScoreWatcher;
import util.listen.training.ZeroGrad;
import util.plot.Plot;
import util.plot.RealTimePlot;

import java.io.File;
import java.util.*;

/**
 * Main class for spiral example. Reimplementation of https://github.com/rtqichen/torchdiffeq/blob/master/examples/latent_ode.py
 */
class Main {

    private static final Logger log = LoggerFactory.getLogger(Main.class);

    @Parameter(names = "-trainBatchSize", description = "Batch size to use for training")
    private int trainBatchSize = 1000;

    @Parameter(names = "-nrofTimeStepsForTraining", description = "Number of time steps per spiral when training")
    private int nrofTimeStepsForTraining = 100;

    @Parameter(names = "-noiseSigma", description = "How much noise to add to generated spirals")
    private double noiseSigma = 0.3;

    @Parameter(names = "-nrofLatentDims", description = "Number of latent dimensions to use")
    private long nrofLatentDims = 4;

    private ComputationGraph model;
    private String modelName;
    private SpiralIterator iterator;
    private Plot<Double, Double> reconstructionPlot;

    public static void main(String[] args) {
        ch.qos.logback.classic.Logger root = (ch.qos.logback.classic.Logger) LoggerFactory.getLogger(Logger.ROOT_LOGGER_NAME);
        root.setLevel(Level.INFO);

        final Main main = parseArgs(args);

        main.addListeners();
        main.run();
    }

    private static Main parseArgs(String[] args) {

        final Main main = new Main();
        final Map<String, ModelFactory> modelCommands = new HashMap<>();
        modelCommands.put("odenet", new OdeNetModel());

        JCommander.Builder parbuilder = JCommander.newBuilder()
                .addObject(main);

        for (Map.Entry<String, ModelFactory> command : modelCommands.entrySet()) {
            parbuilder.addCommand(command.getKey(), command.getValue());
        }

        JCommander jCommander = parbuilder.build();
        jCommander.parse(args);

        final ModelFactory factory = modelCommands.get(jCommander.getParsedCommand());

        main.init(
                factory.create(main.nrofTimeStepsForTraining, main.noiseSigma, main.nrofLatentDims),
                factory.name(),
                factory.getPreProcessor());
        return main;
    }

    private void init(ComputationGraph model, String modelName, MultiDataSetPreProcessor preProcessor) {
        this.model = model;
        this.modelName = modelName;
        long cnt = 0;
        for (GraphVertex vertex : model.getVertices()) {
            log.trace("vertex: " + vertex.getVertexName() + " nrof params: " + vertex.numParams());
            cnt += vertex.numParams();
        }
        log.info("Nrof parameters in model: " + cnt);

        final SpiralFactory spiralFactory = new SpiralFactory(0, 0.3, 0, 6 * Math.PI, 1000);
        this.iterator = new SpiralIterator(
                new SpiralIterator.Generator(spiralFactory, noiseSigma, nrofTimeStepsForTraining, new Random(666)),
                trainBatchSize);
        iterator.setPreProcessor(preProcessor);
    }

    private void addListeners() {
        final File savedir = new File("savedmodels" + File.separator + "spiral" + File.separator + modelName);
        log.info("Models will be saved in: " + savedir.getAbsolutePath());
        savedir.mkdirs();

        setupOutputPlotting(savedir);

        final Plot<Integer, Double> meanAndLogVarPlot = new RealTimePlot<>("Mean and log(var) of z", savedir.getAbsolutePath());
        this.reconstructionPlot = new RealTimePlot<>("Reconstruction", savedir.getAbsolutePath());

        model.addListeners(
                new ZeroGrad(),
                new PerformanceListener(1, true),
                new CheckpointListener.Builder(savedir.getAbsolutePath())
                        .keepLast(1)
                        .deleteExisting(true)
                        .saveEveryNIterations(20)
                        .build(),
                new NanScoreWatcher(() -> {
                    throw new IllegalStateException("NaN score!");
                }),
                // Get names from model factory instead?
                new PlotActivations(meanAndLogVarPlot, "qz0_mean", nrofLatentDims),
                new PlotActivations(meanAndLogVarPlot, "qz0_logvar", nrofLatentDims));

    }

    private void setupOutputPlotting(File savedir) {
        // Runnable stuff is because xychart is not thread safe and generates NPEs if one tries to plot
        // to more than one time series at the time
        final List<Runnable> plotInits = new ArrayList<>();
        final Plot<Double, Double> outputPlot = new RealTimePlot<>("Training Output", savedir.getAbsolutePath());
        for (int batchNrToPlot = 0; batchNrToPlot < Math.min(trainBatchSize, 4); batchNrToPlot++) {
            plotInits.add(initTrainingPlot(outputPlot, batchNrToPlot));
            model.addListeners(new PlotDecodedOutput(outputPlot, "decodedOutput", batchNrToPlot));
        }
        for(Runnable r: plotInits) {
            r.run();
        }
    }

    private Runnable initTrainingPlot(Plot<Double, Double> outputPlot, int batchNrToPlot) {
        final String label = "True output " + batchNrToPlot;
        outputPlot.createSeries(label);
        return new Runnable() {
            @Override
            public void run() {
                final INDArray toPlot = iterator.next().getLabels(0);
                for (long i = 0; i < toPlot.size(2); i++) {
                    outputPlot.plotData(label, toPlot.getDouble(batchNrToPlot, 0, i), toPlot.getDouble(batchNrToPlot, 1, i));
                }
            }
        };
    }


    private void run() {
        for (int i = 0; i < 2000; i++) {
            model.fit(iterator.next());

            if (i > 0 && i % 100 == 0) {
                drawSample();
            }
        }
    }

    private void drawSample() {
        log.info("Sampling model...");

        final int toSample = 0;

        final SpiralIterator.SpiralSet spiralSet = iterator.getCurrent();
        final MultiDataSet mds = spiralSet.getMds();
        final INDArray sample = mds.getFeatures(0);

        new PlotDecodedOutput(reconstructionPlot, "Sampled data", toSample)
                .onForwardPass(model, Collections.singletonMap("Sampled data",
                        sample));

        reconstructionPlot.clearData("True trajectory");
        spiralSet.getSpirals().get(toSample).plotBase(reconstructionPlot, "True trajectory");

        final TimeVae timeVae = new TimeVae(model, "z0", "latentOde");

        final INDArray z0 = timeVae.encode(sample).tensorAlongDimension(0, 1).reshape(1, nrofLatentDims);

        final INDArray tsPos = Nd4j.linspace(0, 2 * Math.PI, 2000);
        //final INDArray tsNeg = Nd4j.linspace(0, -Math.PI, 2000);

        final INDArray zsPos = timeVae.timeDependency(z0, tsPos);
        //final INDArray zsNeg = timeVae.timeDependency(z0, tsNeg);

        final INDArray xsPos = timeVae.decode(zsPos);
        //final INDArray xsNeg = flip(timeVae.decode(zsNeg));

        new PlotDecodedOutput(reconstructionPlot, "Learned trajectory (t < 0)", toSample)
                .onForwardPass(model, Collections.singletonMap("Learned trajectory (t < 0)",
                        xsPos));

        // Seems like forward pass with backwards time does not work
//        new PlotDecodedOutput(reconstructionPlot, "Learned trajectory (t > 0)", toSample)
//                .onForwardPass(model, Collections.singletonMap("Learned trajectory (t > 0)",
//                        xsNeg));
    }

//    private INDArray flip(INDArray array) {
//        final INDArray x = array.tensorAlongDimension(0, 2);
//        final INDArray y = array.tensorAlongDimension(1, 2);
//        return Nd4j.vstack(Nd4j.reverse(x.reshape(x.length())), Nd4j.reverse(y.reshape(y.length()))).reshape(array.shape());
//    }
}
