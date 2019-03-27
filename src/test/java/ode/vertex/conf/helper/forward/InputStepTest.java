package ode.vertex.conf.helper.forward;

import ode.solve.conf.DormandPrince54Solver;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;


/**
 * Test cases for {@link InputStep}
 *
 * @author Christian Skarby
 */
public class InputStepTest extends AbstractHelperConfTest {

    @Override
    OdeHelperForward create() {
        return new InputStep(new DormandPrince54Solver(), 1, false);
    }

    @Override
    INDArray[] createInputs(INDArray input) {
        return new INDArray[] {input, Nd4j.linspace(0, 10, 5)};
    }
}