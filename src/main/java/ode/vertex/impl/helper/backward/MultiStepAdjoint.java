package ode.vertex.impl.helper.backward;

import ode.solve.api.FirstOrderSolver;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;

import java.util.Arrays;

/**
 * {@link OdeHelperBackward} using the adjoint method capable of handling multiple time steps. Gradients will be provided
 * for the last time step only.
 *
 * @author Christian Skarby
 */
public class MultiStepAdjoint implements OdeHelperBackward {

    private final FirstOrderSolver solver;
    private final INDArray time;
    private final int timeIndex;

    public MultiStepAdjoint(FirstOrderSolver solver, INDArray time, int timeIndex) {
        this.solver = solver;
        this.time = time;
        this.timeIndex = timeIndex;
        if(time.length() <= 2 || !time.isVector()) {
            throw new IllegalArgumentException("time must be a vector of size > 2! Was of shape: " + Arrays.toString(time.shape())+ "!");
        }
        assertSorted(time);
    }

    private void assertSorted(INDArray time) {
        int signDiffSum = 0;
        for(int i = 0; i < time.length()-1; i++) {
            signDiffSum += Transforms.sign(time.getScalar(i).sub(time.getScalar(i+1))).getDouble(0);
        }

        if(Math.abs(signDiffSum)+1 != time.length()) {
           throw new IllegalArgumentException("Time must be in ascending or descending order! Got: " + time);
        }
    }

    @Override
    public Pair<Gradient, INDArray[]> solve(ComputationGraph graph, InputArrays input, MiscPar miscPars) {
        final INDArray zt = alignInShapeToTimeFirst(input.getLastOutput());
        final INDArray dL_dzt = alignInShapeToTimeFirst(input.getLossGradient());
        final INDArray dL_dzt_time = alignInShapeToTimeFirst(input.getLossGradientTime().dup());

        assertSizeVsTime(zt);
        assertSizeVsTime(dL_dzt);

        final INDArrayIndex[] timeIndexer = createIndexer(time);
        timeIndexer[1] = NDArrayIndex.interval(time.length()-2, time.length());
        final INDArrayIndex[] ztIndexer = createIndexer(input.getLastOutput());
        final INDArrayIndex[] dL_dztIndexer= createIndexer(input.getLossGradient());

        Pair<Gradient, INDArray[]> gradients = null;
        final INDArray timeGradient = Nd4j.zeros(time.shape());
        double lastTime = 0;

        // Go backwards in time
        for (int step = (int)time.length()-1; step > 0; step--) {
            final INDArray ztStep = getStep(ztIndexer,zt, step);
            final INDArray dL_dzt_timeStep = getStep(dL_dztIndexer, dL_dzt_time, step);

            final INDArray dL_dztStep = getStep(dL_dztIndexer, dL_dzt, step);
            if(gradients != null) {
                dL_dztStep.addi(gradients.getSecond()[0]);
            }

            final InputArrays stepInput = new InputArrays(
                    input.getLastInputs(),
                    ztStep,
                    dL_dztStep,
                    dL_dzt_timeStep,
                    input.getRealGradientView()
            );
            timeIndexer[1] = NDArrayIndex.interval(step - 1, step+1);
            final OdeHelperBackward stepSolve = new SingleStepAdjoint(solver, time.get(timeIndexer), timeIndex);
            gradients = stepSolve.solve(graph, stepInput, miscPars);

            if(timeIndex != -1) {
                timeGradient.put(timeIndexer, gradients.getSecond()[timeIndex]);
                lastTime -= timeGradient.get(timeIndexer).getDouble(1);
            }
        }

        if(timeIndex != -1 && gradients != null) {
            timeIndexer[1] = NDArrayIndex.point(0);
            timeGradient.put(timeIndexer, lastTime);
            gradients.getSecond()[timeIndex] = timeGradient;
        }

        gradients.getSecond()[0].addi(getStep(dL_dztIndexer, dL_dzt, 0));

        return gradients;
    }

    private void assertSizeVsTime(INDArray array) {
        if(array.size(0) != time.length()) {
            throw new IllegalArgumentException("Must have same number of in first dimension as there are time steps! Input: "
            + array.size(0) + ", time: " + time.length());
        }
    }

    private INDArrayIndex[] createIndexer(INDArray array) {
        final INDArrayIndex[] indexer = new INDArrayIndex[array.rank()];
        for(int dim = 0; dim < indexer.length; dim++) {
            indexer[dim] = NDArrayIndex.all();
        }
        return indexer;
    }

    private INDArray getStep(INDArrayIndex[] indexer, INDArray array, int step) {
        indexer[0] = NDArrayIndex.point(step);
        return array.get(indexer);
    }

    private INDArray alignInShapeToTimeFirst(INDArray array) {

        final long[] shape = array.shape();
        switch (shape.length) {
            case 3: // Assume recurrent output
                return array.permute(2,0,1);
            case 5: // Assume conv 3D output
                return array.permute(1,0,2,3,4);
            // Should not happen as conf throws exception for other types
            default: throw new UnsupportedOperationException("Rank not supported: " + array.rank());
        }
    }
}
