package util.listen.training;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.BaseTrainingListener;

/**
 * Sets GradientViewArray to zero between iterations
 *
 * @author Christian Skarby
 */
public class ZeroGrad extends BaseTrainingListener {

    @Override
    public void iterationDone(Model model, int iteration, int epoch) {
        model.getGradientsViewArray().assign(0);
    }
}
