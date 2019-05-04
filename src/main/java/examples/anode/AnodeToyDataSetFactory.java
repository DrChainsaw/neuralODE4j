package examples.anode;

import com.beust.jcommander.Parameter;
import lombok.AllArgsConstructor;
import lombok.Getter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.stream.IntStream;

/**
 * Create {@link DataSetIterator}s for g(x) and separable function in section 4.1 of https://arxiv.org/pdf/1904.01681.pdf
 *
 * @author Christian Skarby
 */
public class AnodeToyDataSetFactory {

    private static final Logger log = LoggerFactory.getLogger(AnodeToyDataSetFactory.class);

    @Parameter(names = "-trainBatchSize", description = "Batch size to use for training")
    private int batchSize = 64;

    @Parameter(names = "-nrofExamples", description = "Number of examples to train on")
    private int nrofExamples = 3000;

    @Parameter(names = "-2D", description = "2D toy example will be used when set")
    private boolean use2D = false;

    @Parameter(names = "-separable", description = "A separable data set is created if present")
    private boolean separable = false;

    private static final double r1 = 0.5;
    private static final double r2 = 1.0;
    private static final double r3 = 1.5;

    /**
     * Hold {@link DataSetIterator}s for training and test
     */
    @AllArgsConstructor
    @Getter
    static class DataSetIters {
        private final DataSetIterator train;
        private final DataSetIterator test;
        private final String name;
    }


    public DataSetIters create() {
        final Random rng = Nd4j.getRandomFactory().getNewRandomInstance(123);
        if (use2D && separable) {
            log.info("Create separable 2D data set");
            return create(createSeparable2D(rng));
        }

        if (separable) {
            log.info("Create separable 1D data set");
            return create(createSeparable1D(rng));
        }

        if (use2D) {
            log.info("Create non-separable 2D data set");
            return create(createNonSeparable2D(rng));
        }

        log.info("Create non-separable 1D data set");
        return create(createNonSeparable1D(rng));
    }

    private DataSetIters create(DataSet ds) {
        return new DataSetIters(
                new ViewIterator(ds, batchSize),
                new ViewIterator(ds, nrofExamples),
                (separable ? "separable" : "non-separable") + (use2D ? "_2D" : "_1D"));
    }

    private DataSet createNonSeparable1D(Random rng) {
        final int nrofFirst = nrofExamples / 3;
        final int nrofSecond = nrofExamples - nrofFirst;

        // Mix up examples evenly
        INDArray inds = Nd4j.create(IntStream.range(0, nrofFirst + 1) // +1 to make sure length is >= nrofExamples
                .flatMap(i -> IntStream.of(i, 2 * i + nrofFirst, 2 * i + nrofFirst + 1))
                .mapToDouble(i -> i)
                .toArray()).get(NDArrayIndex.interval(0, nrofExamples));

        // vstack fails with CUDA backend! Therefore, use hstack and transpose
        return new DataSet(
                Nd4j.hstack(
                        Nd4j.rand(1, nrofFirst, -r1, r1, rng),
                        Nd4j.rand(1, nrofSecond, r2, r3, rng)
                                .muli(Transforms.sign(Nd4j.randn(1, nrofSecond, rng)))).transposei()
                        .get(inds).reshape(nrofExamples, 1),
                Nd4j.hstack(
                        Nd4j.ones(1, nrofFirst).negi(),
                        Nd4j.ones(1, nrofSecond)).transposei()
                        .get(inds).reshape(nrofExamples, 1)
        );
    }

    private DataSet createNonSeparable2D(Random rng) {
        final DataSet dists = createNonSeparable1D(rng);

        final INDArray theta = Nd4j.rand(dists.numExamples(), 1, 0, 2 * Math.PI, rng);

        return new DataSet(
                Nd4j.hstack(Transforms.cos(theta).muli(dists.getFeatures()), Transforms.sin(theta).muli(dists.getFeatures())),
                dists.getLabels());
    }

    private DataSet createSeparable1D(Random rng) {
        final int nrofFirst = nrofExamples / 2;
        final int nrofSecond = nrofExamples - nrofFirst;

        // Mix up examples evenly
        INDArray inds = Nd4j.create(IntStream.range(0, nrofFirst + 1) // +1 to make sure length is >= nrofExamples
                .flatMap(i -> IntStream.of(i, i + nrofFirst))
                .mapToDouble(i -> i)
                .toArray()).get(NDArrayIndex.interval(0, nrofExamples));

        // vstack fails with CUDA backend! Therefore, use hstack and transpose
        return new DataSet(
                Nd4j.hstack(
                        Nd4j.rand(1, nrofFirst, 0, r1, rng),
                        Nd4j.rand(1, nrofSecond, -r3, -r2, rng)).transposei()
                        .get(inds).reshape(nrofExamples, 1),
                Nd4j.hstack(
                        Nd4j.ones(1, nrofFirst).negi(),
                        Nd4j.ones(1, nrofSecond)).transposei()
                        .get(inds).reshape(nrofExamples, 1)
        );
    }


    private DataSet createSeparable2D(Random rng) {
        final DataSet noise = createSeparable1D(rng);

        final INDArray theta = Nd4j.rand(noise.numExamples(), 1, -Math.PI, Math.PI, rng);

        return new DataSet(
                Nd4j.hstack(theta, Transforms.sin(theta).addi(noise.getFeatures())),
                noise.getLabels()
        );
    }

    public static void main(String[] args) {
        final AnodeToyDataSetFactory fac = new AnodeToyDataSetFactory();

        final DataSet dsn1D = fac.createNonSeparable1D(Nd4j.getRandomFactory().getNewRandomInstance(666));
        PlotSteps3D.plotXYZ(dsn1D.getFeatures(), dsn1D.getLabels(), new ScatterPlot3D("1D non-separable", ""));

        final DataSet dsn2D = fac.createNonSeparable2D(Nd4j.getRandomFactory().getNewRandomInstance(666));
        PlotSteps3D.plotXYZ(dsn2D.getFeatures(), dsn2D.getLabels(), new ScatterPlot3D("2D non-separable", ""));

        final DataSet ds1D = fac.createSeparable1D(Nd4j.getRandomFactory().getNewRandomInstance(666));
        PlotSteps3D.plotXYZ(ds1D.getFeatures(), ds1D.getLabels(), new ScatterPlot3D("1D separable", ""));

        final DataSet ds2D = fac.createSeparable2D(Nd4j.getRandomFactory().getNewRandomInstance(666));
        PlotSteps3D.plotXYZ(ds2D.getFeatures(), ds2D.getLabels(), new ScatterPlot3D("2D separable", ""));
    }
}
