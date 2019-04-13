package ode.vertex.impl.helper.backward.timegrad;

import ode.solve.api.FirstOrderEquation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class NoTimeGrad implements TimeGrad {

    private final static TimeGrad noTimeGrad = new NoTimeGrad();
    public static TimeGrad.Factory factory = new TimeGrad.Factory() {

        @Override
        public TimeGrad create() {
            // Singleton because stateless
            return noTimeGrad;
        }
    };


    @Override
    public INDArray calcTimeAdjointT1(FirstOrderEquation equation, INDArray zt1, INDArray time) {
        return Nd4j.empty();
    }

    @Override
    public INDArray[] createLossGradient(INDArray zAdjoint, INDArray tAdjoint) {
        return new INDArray[] {zAdjoint};
    }
}
