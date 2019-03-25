package ode.vertex.conf.helper.backward;

import ode.solve.conf.DormandPrince54Solver;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Test cases for {@link InputStepAdjoint}
 *
 * @author Christian Skarby
 */
public class InputStepAdjointTest extends AbstractHelperConfTest {

    @Override
    OdeHelperBackward create(int nrofTimeSteps, boolean needTimeGrad) {
        return new InputStepAdjoint(new DormandPrince54Solver(), 1, needTimeGrad);
    }

    @Override
    INDArray[] createInputs(INDArray input, int nrofTimeSteps) {
        return new INDArray[]{input, Nd4j.linspace(0, 2, nrofTimeSteps)};
    }
}