package examples.anode;

import ode.solve.api.StepListener;
import ode.solve.impl.util.SolverState;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;
import util.plot.Plot;

import java.awt.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Plots the state of each solver step
 *
 * @author Christian Skarby
 */
class PlotState implements StepListener {

    private final Plot<Double, Double> plot;
    private final INDArray labels;

    PlotState(Plot<Double, Double> plot, INDArray labels) {
        this.plot = plot;
        this.labels = labels;
    }

    @Override
    public void begin(INDArray t, INDArray y0) {
        plot.clearData();
        plotXY(y0);
    }

    @Override
    public void step(SolverState solverState, INDArray step, INDArray error) {
        plotXY(solverState.getCurrentState());
    }

    @Override
    public void done() {

    }

    private void plotXY(INDArray xyLoc) {
        plot.createSeries("g(x) = 1").scatter().set(Color.blue);
        plot.createSeries("g(x) = -1").scatter().set(Color.red);

        final double[] x;
        final double[] y;
        if (xyLoc.size(1) == 1) {
            y = xyLoc.toDoubleVector();
            x = new double[y.length]; // Init to zeros
        } else {
            x = xyLoc.getColumn(0).toDoubleVector();
            y = xyLoc.getColumn(1).toDoubleVector();
        }
        final int[] labs = labels.toIntVector();

        final Map<Integer, Pair<List<Double>, List<Double>>> data = new HashMap<>();
        data.put(1, new Pair<>(new ArrayList<>(), new ArrayList<>()));
        data.put(-1, new Pair<>(new ArrayList<>(), new ArrayList<>()));

        for (int i = 0; i < xyLoc.size(0); i++) {
            data.get(labs[i]).getFirst().add(x[i]);
            data.get(labs[i]).getSecond().add(y[i]);
        }

        plot.plotData("g(x) = 1", data.get(1).getFirst(), data.get(1).getSecond());
        plot.plotData("g(x) = -1", data.get(-1).getFirst(), data.get(-1).getSecond());
    }
}
