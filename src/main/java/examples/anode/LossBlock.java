package examples.anode;

import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.lossfunctions.impl.LossMSE;

/**
 * Adds loss layer for ANODE experiments
 *
 * @author Christian Skarby
 */
public class LossBlock implements Block {

    @Override
    public String add(GraphBuilderWrapper builder, String... prev) {
        builder.addLayer("output", new OutputLayer.Builder()
                .nOut(1)
                .activation(new ActivationIdentity())
                .lossFunction(new LossMSE())
                .build(), prev);

        return "output";
    }
}
