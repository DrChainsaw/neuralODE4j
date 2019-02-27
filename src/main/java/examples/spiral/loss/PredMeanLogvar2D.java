package examples.spiral.loss;

import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.primitives.Triple;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Extracts prediction, mean and log(var) from a 2D variable when given the number of latent dimensions
 *
 * @author Christian Skarby
 */
@Data
public class PredMeanLogvar2D implements NormElboLoss.ExtractQzZero {

    private final long nrofLatentDims;

    public PredMeanLogvar2D(@JsonProperty("nrofLatentDims") long nrofLatentDims) {
        this.nrofLatentDims = nrofLatentDims;
    }

    @Override
    public Triple<INDArray, INDArray, INDArray> extractPredMeanLogvar(INDArray result) {
        final long predSize = result.size(1) - 2 * nrofLatentDims;
        return new Triple<>(
                // Here we "unpack" the result of the merge above.
                result.get(NDArrayIndex.all(), NDArrayIndex.interval(0, predSize)),
                result.get(NDArrayIndex.all(), NDArrayIndex.interval(predSize, predSize + nrofLatentDims)),
                result.get(NDArrayIndex.all(), NDArrayIndex.interval(predSize + nrofLatentDims, predSize + 2 * nrofLatentDims))
        );
    }

    @Override
    public INDArray combinePredMeanLogvarEpsilon(INDArray predEps, INDArray meanEps, INDArray logvarEps) {
        return Nd4j.hstack(
                predEps,
                meanEps,
                logvarEps);
    }
}
