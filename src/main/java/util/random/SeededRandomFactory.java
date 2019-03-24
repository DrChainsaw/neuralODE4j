package util.random;

import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.RandomFactory;

/**
 * {@link RandomFactory} with a configurable random seed
 *
 * @author Christian Skarby
 */
public class SeededRandomFactory extends RandomFactory {

    final java.util.Random base;
    private ThreadLocal<Random> threadRandom = new ThreadLocal<>();

    public SeededRandomFactory(Class randomClass, long baseSeed) {
        super(randomClass);
        base = new java.util.Random(baseSeed);
    }

    @Override
    public Random getRandom() {
        // Copy pase from RandomFactory
        try {
            if (threadRandom.get() == null) {
                Random t = super.getNewRandomInstance(base.nextLong());
                threadRandom.set(t);
                return t;
            }

            return threadRandom.get();
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }

    @Override
    public Random getNewRandomInstance() {
        return super.getNewRandomInstance(base.nextLong());
    }

    /**
     * Set a base seed for all Nd4j random generators
     * @param seed base seed
     */
    public static void setNd4jSeed(long seed) {
        Nd4j.randomFactory = new SeededRandomFactory(Nd4j.randomFactory.getRandom().getClass(), seed);
    }
}
