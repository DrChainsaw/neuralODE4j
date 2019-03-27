package ode.vertex.conf.helper.backward;

import ode.solve.conf.DormandPrince54Solver;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Test cases for {@link FixedStepAdjoint}
 *
 * @author Christian Skarby
 */
public class FixedStepAdjointTest extends AbstractHelperConfTest {

    @Override
    OdeHelperBackward create(int nrofTimeSteps, boolean needTimeGradient) {
        return new FixedStepAdjoint(new DormandPrince54Solver(), Nd4j.linspace(0,3,nrofTimeSteps));
    }

    @Override
    INDArray[] createInputs(INDArray input, int nrofTimeSteps) {
        return new INDArray[] {input};
    }
}