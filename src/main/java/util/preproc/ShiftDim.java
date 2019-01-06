package util.preproc;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Random;
import java.util.stream.Stream;

/**
 * Shifts the feature array in the given dimension
 *
 * @author Christian Skarby
 */
public class ShiftDim implements DataSetPreProcessor {

    private final int dimension;
    private final ShiftSupplier shiftSupplier;

    /**
     * Interface for providing policy for shifting
     */
    public interface ShiftSupplier {
        /**
         * Get next shift
         * @return number of elements to shift
         */
        long nextShift();
    }

    public ShiftDim(int dimension, Random rng, final int bound) {
        this(dimension, () -> bound - rng.nextInt(2*bound));
    }

    public ShiftDim(int dimension, ShiftSupplier shiftSupplier) {
        this.dimension = dimension;
        this.shiftSupplier = shiftSupplier;
    }

    @Override
    public void preProcess(DataSet toPreProcess) {
        INDArray features = toPreProcess.getFeatures();

        if(features.isVector()) {
            features = features.reshape(features.length());
        }

        final INDArrayIndex[] shiftGet = Stream.generate(NDArrayIndex::all)
                .limit(features.rank())
                .toArray(INDArrayIndex[]::new);
        final INDArrayIndex[] shiftPut = shiftGet.clone();
        final INDArrayIndex[] zeros = shiftGet.clone();

        final long sizeDim = features.size(dimension);

        if(setShift(shiftGet, shiftPut, zeros, sizeDim)) {
            features.put(shiftPut, features.get(shiftGet).dup()); // dup seems to be needed with CPU backend
            features.put(zeros, 0);
        }
    }

    private boolean setShift(INDArrayIndex[] shiftGet, INDArrayIndex[] shiftPut, INDArrayIndex[] zeros, long sizeDim) {
        final long nextShift = shiftSupplier.nextShift();
        if(nextShift == 0) {
            return false;
        }

        if(nextShift > 0) {
            shiftGet[dimension] = NDArrayIndex.interval(0, sizeDim - nextShift);
            shiftPut[dimension] = NDArrayIndex.interval(nextShift,  sizeDim);
            zeros[dimension] = NDArrayIndex.interval(0, nextShift);
        } else {
            shiftGet[dimension] = NDArrayIndex.interval(-nextShift, sizeDim);
            shiftPut[dimension] = NDArrayIndex.interval(0,  sizeDim + nextShift);
            zeros[dimension] = NDArrayIndex.interval(sizeDim + nextShift, sizeDim);
        }
        return true;
    }
}
