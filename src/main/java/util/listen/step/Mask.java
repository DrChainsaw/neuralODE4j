package util.listen.step;

import ode.solve.api.StepListener;
import ode.solve.impl.util.SolverState;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Masks step from a given listener based on a user specified criterion or using one of the built in criteria
 * (forward, backward)
 *
 * @author Christian Skarby
 */
public class Mask implements StepListener {

    private final StepListener listener;
    private final MaskFunction maskFunction;
    private boolean mask = false;

    /**
     * Mask when solving in forward direction
     * @param listener Listener to mask calls to
     * @return Mask which masks the forward direction
     */
    public static Mask forward(StepListener listener) {
        return new Mask(listener, new MaskFunction() {
            @Override
            public boolean shallMask(INDArray t, INDArray y0) {
                return t.argMax().getInt(0) == 0;
            }
        });
    }

    /**
     * Mask when solving in forward direction
     * @param listener Listener to mask calls to
     * @return Mask which masks the forward direction
     */
    public static Mask backward(StepListener listener) {
        return new Mask(listener, new MaskFunction() {
            @Override
            public boolean shallMask(INDArray t, INDArray y0) {
                return t.argMax().getInt(0) == 1;
            }
        });
    }

    public interface MaskFunction {
        boolean shallMask(INDArray t, INDArray y0);
    }

    public Mask(StepListener listener, MaskFunction maskFunction) {
        this.listener = listener;
        this.maskFunction = maskFunction;
    }

    @Override
    public void begin(INDArray t, INDArray y0) {
        mask = maskFunction.shallMask(t, y0);
        if(mask) {
            listener.begin(t, y0);
        }
    }

    @Override
    public void step(SolverState solverState, INDArray step, INDArray error) {
        if(mask) {
            listener.step(solverState, step, error);
        }
    }

    @Override
    public void done() {
        if(mask) {
            listener.done();
        }
    }
}
