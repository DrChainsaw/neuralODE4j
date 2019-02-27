package examples.spiral;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.CompositeMultiDataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;

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
    private MultiDataSet current;
    private MultiDataSetPreProcessor preProcessor = new CompositeMultiDataSetPreProcessor(); // Noop

    /**
     * Generates {@link MultiDataSet}s from a {@link SpiralFactory}.
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

        MultiDataSet generate(int batchSize) {
            final List<SpiralFactory.Spiral> spirals = factory.sample(
                    batchSize,
                    nrofSamples,
                    () -> Math.min(0.9, Math.max(0.1,rng.nextDouble())),
                    rng::nextBoolean);

            final INDArray trajFeature = Nd4j.createUninitialized(new long[] {batchSize, 2, nrofSamples});
            final INDArray tFeature = spirals.get(0).theta().dup();
            tFeature.subi(tFeature.minNumber());
            for(int i = 0; i < batchSize; i++) {
                trajFeature.tensorAlongDimension(i, 1,2).assign(spirals.get(i).trajectory());
            }
            trajFeature.addi(Nd4j.randn(trajFeature.shape(), Nd4j.getRandomFactory().getNewRandomInstance(rng.nextLong())).muli(noiseSigma));
            return new org.nd4j.linalg.dataset.MultiDataSet(
                    new INDArray[] {trajFeature, tFeature},
                    new INDArray[] {trajFeature});
        }
    }

    public SpiralIterator(Generator generator, int batchSize) {
        this.generator = generator;
        this.batchSize = batchSize;
    }

    @Override
    public MultiDataSet next(int num) {
        if(current == null || num != current.getFeatures(0).size(0)) {
            current = generator.generate(num);
            preProcessor.preProcess(current);
        }
        return current;
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

}
