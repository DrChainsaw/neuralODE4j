package examples.spiral.loss;

import lombok.Data;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.primitives.Triple;
import org.nd4j.shade.jackson.annotation.JsonProperty;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

import java.io.Serializable;

/**
 * Negative Evidence Lower BOund (ELBO) under assumption that the distribution is normal.
 * Reimplementation of https://github.com/rtqichen/torchdiffeq/blob/master/examples/latent_ode.py
 *
 * @author Christian Skarby
 */
@Data
public class NormElboLoss implements ILossFunction{

    private final static double log2pi = Math.log(2 * Math.PI);

    private final ILossFunction reconstructionLoss;
    private final ILossFunction kldLoss;
    private final ExtractQzZero extract;

    @JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "@class")
    public interface ExtractQzZero extends Serializable {

        Triple<INDArray, INDArray, INDArray> extractPredMeanLogvar(INDArray result);
        INDArray combinePredMeanLogvarEpsilon(INDArray predEps, INDArray meanEps, INDArray logvarEps);
    }

    public NormElboLoss(
            @JsonProperty("reconstructionLoss") ILossFunction reconstructionLoss,
            @JsonProperty("kldLoss") ILossFunction kldLoss,
            @JsonProperty("extract") ExtractQzZero extract) {
        this.reconstructionLoss = reconstructionLoss;
        this.kldLoss = kldLoss;
        this.extract = extract;
    }

    @Override
    public double computeScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {
        if(average) {
            return scoreArray(labels, preOutput, activationFn, mask).meanNumber().doubleValue();
        }
        return scoreArray(labels, preOutput, activationFn, mask).sumNumber().doubleValue();
    }

    @Override
    public INDArray computeScoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        return scoreArray(labels, preOutput, activationFn, mask);
    }

    @Override
    public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {

        final Triple<INDArray, INDArray, INDArray> predMeanVar = extract.extractPredMeanLogvar(preOutput);

        if (!labels.equalShapes(predMeanVar.getFirst())) {
            Preconditions.throwEx("Labels and preOutput must have equal shapes: got shapes %s vs %s", labels.shape(), predMeanVar.getFirst().shape());
        }

        final INDArray predGrad = reconstructionLoss.computeGradient(labels, predMeanVar.getFirst(), activationFn, mask);
        final INDArray meanAndLogVar =  Nd4j.hstack(predMeanVar.getSecond(), predMeanVar.getThird());
        final INDArray meanAndLogvarGrad = kldLoss.computeGradient(Nd4j.zeros(meanAndLogVar.shape()), meanAndLogVar, activationFn, mask);
        final INDArray meanGrad = meanAndLogvarGrad.get(NDArrayIndex.all(),
                NDArrayIndex.interval(0, predMeanVar.getSecond().size(1)));
        final INDArray logvarGrad = meanAndLogvarGrad.get(NDArrayIndex.all(),
                NDArrayIndex.interval(predMeanVar.getSecond().size(1), meanAndLogvarGrad.size(1)));

        return extract.combinePredMeanLogvarEpsilon(predGrad, meanGrad, logvarGrad);
    }

    @Override
    public Pair<Double, INDArray> computeGradientAndScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {
        return new Pair<>(
                computeScore(labels, preOutput, activationFn, mask, average),
                computeGradient(labels, preOutput, activationFn, mask));
    }

    private INDArray scoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {

        final Triple<INDArray, INDArray, INDArray> predMeanVar = extract.extractPredMeanLogvar(preOutput);

        if (!labels.equalShapes(predMeanVar.getFirst())) {
            Preconditions.throwEx("Labels and preOutput must have equal shapes: got shapes %s vs %s", labels.shape(), predMeanVar.getFirst().shape());
        }
        final INDArray reconstructionScore = reconstructionLoss.computeScoreArray(labels, predMeanVar.getFirst(), activationFn, mask);
        final INDArray meanAndLogVar =  Nd4j.hstack(predMeanVar.getSecond(), predMeanVar.getThird());
        final INDArray meanAndLogvarScore = kldLoss.computeScoreArray(
                Nd4j.zeros(meanAndLogVar.shape()),
                meanAndLogVar,
                activationFn,
                mask);

        return reconstructionScore.add(meanAndLogvarScore);
    }

    @Override
    public String name() {
        return "NormalElboLoss(" + reconstructionLoss + ", "+ kldLoss + ")";
    }
}
