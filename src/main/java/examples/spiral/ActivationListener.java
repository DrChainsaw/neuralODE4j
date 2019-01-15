package examples.spiral;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Map;

/**
 * Listens to activations of a given layer
 *
 * @author Christian Skarby
 */
public class ActivationListener extends BaseTrainingListener {

    private final String layerToListen;
    private INDArray lastActivation;

    public ActivationListener(String layerToListen) {
        this.layerToListen = layerToListen;
    }

    @Override
    public void onForwardPass(Model model, Map<String, INDArray> activations) {
        lastActivation = activations.get(layerToListen).detach();
    }

    INDArray lastActivation() {
        return lastActivation;
    }
}
