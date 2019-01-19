package ode.vertex.conf.helper.forward;

import ode.solve.conf.DormandPrince54Solver;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Test cases for {@link FixedStep}
 *
 * @author Christian Skarby
 */
public class FixedStepTest extends AbstractHelperConfTest{


    @Override
    OdeHelperForward create() {
        return new FixedStep(new DormandPrince54Solver(), Nd4j.linspace(0,3,4));
    }

    @Override
    INDArray[] createInputs(INDArray input) {
        return new INDArray[] {input};
    }
}