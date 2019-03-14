package examples.spiral;

import lombok.AllArgsConstructor;
import lombok.Getter;
import org.deeplearning4j.util.TimeSeriesUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.CompositeMultiDataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * {@link MultiDataSetIterator} for spirals. Note that the same {@link MultiDataSet} instance will be used until
 * reset is called. This is what the original implementation does as well
 *
 * @author Christian Skarby
 */
public class SpiralIterator implements MultiDataSetIterator {

    private final Generator generator;
    private final int batchSize;
    private SpiralSet current;
    private MultiDataSetPreProcessor preProcessor = new CompositeMultiDataSetPreProcessor(); // Noop

    @Getter @AllArgsConstructor
    public static class SpiralSet {
        private final MultiDataSet mds;
        private final List<SpiralFactory.Spiral> spirals;
    }

    /**
     * Generates {@link SpiralSet}s from a {@link SpiralFactory}.
     */
    public static class Generator {
        private final SpiralFactory factory;
        private final double noiseSigma;
        private final long nrofSamples;
        private final Random rng;


        public Generator(SpiralFactory factory, double noiseSigma, long nrofSamples, Random rng) {
            this.factory = factory;
            this.noiseSigma = noiseSigma;
            this.nrofSamples = nrofSamples;
            this.rng = rng;
        }

        SpiralSet generate(int batchSize) {

            final double sampoffset = nrofSamples / (double)factory.baseNrofSamples();
            final double samprange = 1 - 2*sampoffset;

            final List<SpiralFactory.Spiral> spirals = factory.sample(
                    batchSize,
                    nrofSamples,
                    () -> sampoffset + rng.nextDouble()*samprange,
                    rng::nextBoolean);

            final INDArray trajFeature = Nd4j.createUninitialized( new long[] {batchSize, 2, nrofSamples}, 'f');
            final INDArray tFeature = spirals.get(0).theta().dup('f');
            tFeature.subi(tFeature.minNumber());

            for(int i = 0; i < batchSize; i++) {
                trajFeature.tensorAlongDimension(i, 1,2).assign(spirals.get(i).trajectory());
            }

            trajFeature.addi(Nd4j.randn(trajFeature.shape(), Nd4j.getRandomFactory().getNewRandomInstance(rng.nextLong())).muli(noiseSigma));
            return new SpiralSet(new org.nd4j.linalg.dataset.MultiDataSet(
                    // Reverse trajectory so last time step of RNN represents first element of trajectory.
                    // Unsure if really needed since RNN anyways mangles whole sequence into something (mean and var)
                    new INDArray[] {TimeSeriesUtils.reverseTimeSeries(trajFeature), tFeature},
                    new INDArray[] {trajFeature}),
                    Collections.unmodifiableList(spirals));
        }
    }

    public SpiralIterator(Generator generator, int batchSize) {
        this.generator = generator;
        this.batchSize = batchSize;
    }

    @Override
    public MultiDataSet next(int num) {
        if(current == null || num != current.getMds().getFeatures(0).size(0)) {
            current = generator.generate(num);
            preProcessor.preProcess(current.getMds());
        }
        return current.getMds();
    }

    @Override
    public void setPreProcessor(MultiDataSetPreProcessor preProcessor) {
        this.preProcessor = preProcessor;
    }

    @Override
    public MultiDataSetPreProcessor getPreProcessor() {
        return preProcessor;
    }

    @Override
    public boolean resetSupported() {
        return false;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public void reset() {
        current = null;
    }

    @Override
    public boolean hasNext() {
        return true;
    }

    @Override
    public MultiDataSet next() {
        return next(batchSize);
    }

    public SpiralSet getCurrent() {
        return current;
    }
}
