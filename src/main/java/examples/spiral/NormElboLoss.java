package examples.spiral;

import org.nd4j.base.Preconditions;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.primitives.Triple;

/**
 * Negative Evidence Lower BOund (ELBO) under assumption that the distribution is normal.
 *
 * @author Christian Skarby
 */
public class NormElboLoss implements ILossFunction{

    private final double logNoiseVar;
    private final double logMean;
    private final ExtractQzZero extract;

    public interface ExtractQzZero {

        Triple<INDArray, INDArray, INDArray> extractPredMeanVar(INDArray result);

    }

    public NormElboLoss(double mean, double noiseSigma, ExtractQzZero extract) {
        this.logNoiseVar = 2*Math.log(noiseSigma);
        this.logMean = Math.log(mean);
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
        return null;
    }

    @Override
    public Pair<Double, INDArray> computeGradientAndScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {
        return null;
    }

    private INDArray scoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {

        final INDArray output = activationFn.getActivation(preOutput.dup(), true);
        final Triple<INDArray, INDArray, INDArray> predMeanVar = extract.extractPredMeanVar(output);

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
        return Transforms.pow(output.rsubi(labels), 2, false)
                .divi(Math.exp(logNoiseVar))
                .addi(logMean)
                .addi(logNoiseVar)
                .muli(-0.5);
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

    @Override
    public String name() {
        return "NormalElboLoss";
    }
}
