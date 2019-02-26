package examples.spiral;

import lombok.Data;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.primitives.Triple;
import org.nd4j.shade.jackson.annotation.JsonProperty;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

import java.io.Serializable;

/**
 * Negative Evidence Lower BOund (ELBO) under assumption that the distribution is normal.
 *
 * @author Christian Skarby
 */
@Data
public class NormElboLoss implements ILossFunction{

    private final static double log2pi = Math.log(2 * Math.PI);

    private final double logNoiseVar;
    private final ExtractQzZero extract;

    @JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "@class")
    public interface ExtractQzZero extends Serializable {

        Triple<INDArray, INDArray, INDArray> extractPredMeanLogvar(INDArray result);
        INDArray combinePredMeanLogvarEpsilon(INDArray predEps, INDArray meanEps, INDArray logvarEps);
    }

    public NormElboLoss(@JsonProperty("noiseSigma") double noiseSigma, @JsonProperty("extract") ExtractQzZero extract) {
        this.logNoiseVar = 2*Math.log(noiseSigma);
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
        final INDArray output = activationFn.getActivation(preOutput.dup(), true);
        final Triple<INDArray, INDArray, INDArray> predMeanVar = extract.extractPredMeanLogvar(output);

        if (!labels.equalShapes(predMeanVar.getFirst())) {
            Preconditions.throwEx("Labels and preOutput must have equal shapes: got shapes %s vs %s", labels.shape(), predMeanVar.getFirst().shape());
        }

        final INDArray predGrad = logNormalPdfGradient(labels, predMeanVar.getFirst());
        final INDArray meanGrad = normalKlGradMu1(predMeanVar.getSecond());
        final INDArray logvarGrad = normalKlGradLv1(predMeanVar.getThird());

        return activationFn.backprop(output, extract.combinePredMeanLogvarEpsilon(predGrad, meanGrad, logvarGrad)).getFirst();
    }

    @Override
    public Pair<Double, INDArray> computeGradientAndScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {
        return new Pair<>(
                computeScore(labels, preOutput, activationFn, mask, average),
                computeGradient(labels, preOutput, activationFn, mask));
    }

    private INDArray scoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {

        final INDArray output = activationFn.getActivation(preOutput.dup(), true);
        final Triple<INDArray, INDArray, INDArray> predMeanVar = extract.extractPredMeanLogvar(output);

        final int[] sumDims = new int[predMeanVar.getFirst().rank()-1];
        for(int i = 0; i < sumDims.length; i++) {
            sumDims[i] = i+1;
        }

        if (!labels.equalShapes(predMeanVar.getFirst())) {
            Preconditions.throwEx("Labels and preOutput must have equal shapes: got shapes %s vs %s", labels.shape(), predMeanVar.getFirst().shape());
        }

        final INDArray logPx = logNormalPdf(labels, predMeanVar.getFirst()).sum(sumDims);
        final INDArray analyticKl = normalKl(predMeanVar.getSecond(), predMeanVar.getThird()).sum(1);

        return analyticKl.subi(logPx);
    }


    private INDArray logNormalPdf(INDArray labels, INDArray output) {
        // Expression from original repo.
        // Similar to log-likelihood assuming parameters are from a gaussian distribution, but not 100% same?
        return Transforms.pow(output.rsub(labels), 2, false)
                .divi(Math.exp(logNoiseVar))
                .addi(log2pi)
                .addi(logNoiseVar)
                .muli(-0.5);
    }

    private INDArray logNormalPdfGradient(INDArray labels, INDArray output) {
        // 2 from derivative of exponent and -0.5 constant cancel out
        return output.sub(labels).divi(Math.exp(logNoiseVar));
    }

    private INDArray normalKl(INDArray mu1, INDArray lv1 ) {
        final double mu2 = 0;
        final INDArray v1 = Transforms.exp(lv1);
        final double v2 = 1;
        final INDArray lstd1 = lv1.div(2);
        final double lstd2 = 0;
        return lstd1.rsubi(lstd2).addi(
                v1.addi(Transforms.pow(mu1.sub(mu2), 2, false)).divi(2 * v2)
        ).subi(0.5);
    }

    private INDArray normalKlGradMu1(INDArray mu1) {
        final double mu2 = 0;
        final double v2 = 1;
        return mu1.sub(mu2).muli(v2);
    }

    private INDArray normalKlGradLv1(INDArray lv1) {
        final double lv2 = 0;
        return Transforms.exp(lv1.sub(lv2)).subi(1).divi(2);
    }

    @Override
    public String name() {
        return "NormalElboLoss(" + logNoiseVar +")";
    }
}
