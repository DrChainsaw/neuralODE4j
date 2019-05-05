package examples.anode;

import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.nd4j.linalg.activations.impl.ActivationIdentity;

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
                // Loss function not stated in paper.
                // MSE works, but can not achieve better loss than 2/3 for 1D non-separable ODE vs 1/3 presented in paper
                // Inner set and upper outer set is mapped to 0 while lower outer is mapped to -1.
                // Avg MSE is mean[(1-0)^2 + (-1-0)^2 + (-1 -(-1))^2] = 2/3. L1 loss gives same result due to 1^2 = 1
                // Huber loss perhaps? It is implemented in PyTorch and will give 1/3 loss for the above case
                .lossFunction(new LossHuber())
                .build(), prev);

        return "output";
    }
}
