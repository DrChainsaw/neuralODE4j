package examples.spiral;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * {@link MultiDataSetIterator} for spirals.
 *
 * @author Christian Skarby
 */
public class SpiralIterator implements MultiDataSetIterator {

    private final Generator generator;
    private final int batchSize;
    private MultiDataSet current;

    public static class Generator {
        private final SpiralFactory factory;
        private final double noiseVar;
        private final long nrofSamples;
        private final Random rng;


        public Generator(SpiralFactory factory, double noiseVar, long nrofSamples, Random rng) {
            this.factory = factory;
            this.noiseVar = noiseVar;
            this.nrofSamples = nrofSamples;
            this.rng = rng;
        }

        MultiDataSet generate(int batchSize) {
            final List<SpiralFactory.Spiral> spirals = factory.sample(
                    batchSize,
                    nrofSamples,
                    () -> Math.min(0.9, Math.max(0.1,rng.nextDouble())),
                    rng::nextBoolean);
            final List<INDArray> trajs = new ArrayList<>(batchSize);
            final List<INDArray> ts = new ArrayList<>(batchSize);
            for(SpiralFactory.Spiral spiral: spirals) {
                trajs.add(spiral.trajectory());
                ts.add(spiral.theta());
            }
            final INDArray trajFeature = Nd4j.hstack( trajs.toArray(new INDArray[0])).reshape(batchSize, nrofSamples, 2);
            trajFeature.addi(Nd4j.randn(trajFeature.shape(), Nd4j.getRandomFactory().getNewRandomInstance(rng.nextLong())).muli(noiseVar));
            final INDArray tFeature = Nd4j.concat(1, ts.toArray(new INDArray[0]));
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
        return null;
    }

    @Override
    public void setPreProcessor(MultiDataSetPreProcessor preProcessor) {

    }

    @Override
    public MultiDataSetPreProcessor getPreProcessor() {
        return null;
    }

    @Override
    public boolean resetSupported() {
        return false;
    }

    @Override
    public boolean asyncSupported() {
        return false;
    }

    @Override
    public void reset() {

    }

    @Override
    public boolean hasNext() {
        return false;
    }

    @Override
    public MultiDataSet next() {
        return null;
    }
}
