package ode.vertex.impl.gradview;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * A simple contiguous gradient view
 *
 * @author Christian Skarby
 */
public class Contiguous1DView implements INDArray1DView {

    private final INDArray view;

    public Contiguous1DView(INDArray view) {
        this.view = view;
    }

    @Override
    public void assignFrom(INDArray toAssign) {
        if(toAssign.length() != view.length()) {
            throw new IllegalArgumentException("Array to assignFrom must have same length! " +
                    "This length: " + view.length() +" array length: " + toAssign.length());
        }

        if(toAssign.rank() != 1) {
            throw new IllegalArgumentException("Array toAssign must have rank 1!");
        }

        view.assign(toAssign);
    }

    @Override
    public void assignTo(INDArray assignTo) {
        if(assignTo.length() != view.length()) {
            throw new IllegalArgumentException("Array assignTo must have same length! " +
                    "This length: " + view.length() +" array length: " + assignTo.length());
        }
        if(assignTo.rank() != 1) {
            throw new IllegalArgumentException("Array assignTo must have rank 1!");
        }

        assignTo.assign(view);
    }

    @Override
    public long length() {
        return view.length();
    }
}
