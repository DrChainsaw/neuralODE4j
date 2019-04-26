package examples.anode;

import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.activations.impl.ActivationReLU;

/**
 * Multi Layer Perceptron {@link Block} from section E1.1 in https://arxiv.org/pdf/1904.01681.pdf.
 * <br><br>
 * It is not clear to me what the description "dinput + 1 → dhidden → ReLU → dhidden → ReLU → dinput" actually means.
 * Interpretation made here is the following:
 * <br><br>
 * "dinput + 1 → dhidden → ReLU" => dense layer of dimension dinput + 1 x dhidden followed by RELU activation
 * <br><br>
 * "dhidden → ReLU" => dense layer of dimension dhidden x dhidden followed by RELU activation
 * <br><br>
 * "dinput" => dense layer of dimensions dhidden x dinput (no activation function)
 * <br><br>
 * A more consistent interpretation might be to interpret "dinput + 1" as a separate dense layer of size
 * dinput + 1 x dinput +1, but that would not make much sense.
 * <br><br>
 * The last part; "dhidden → ReLU → dinput", could be interpreted as just a dense layer of dimension dhidden x dinput
 * with RELU but then the model would not be able to output any negative numbers.
 *
 * @author Christian Skarby
 */
public class MlpBlock implements Block {

    private final DenseLayer.Builder outputLayer;
    private final DenseLayer.Builder hiddenLayer;

    public MlpBlock(long nrofHidden, long nrofOutputs) {
        hiddenLayer = new DenseLayer.Builder()
                .nOut(nrofHidden)
                .activation(new ActivationReLU());
        outputLayer = new DenseLayer.Builder()
                .nOut(nrofOutputs)
                .activation(new ActivationIdentity());
    }

    @Override
    public String add(GraphBuilderWrapper builder, String... prev) {
        builder
                .addLayer("h1", hiddenLayer.build(), prev)
                .addLayer("h2", hiddenLayer.build(), "h1")
                .addLayer("out", outputLayer.build(), "h2");
        return "out";
    }
}
