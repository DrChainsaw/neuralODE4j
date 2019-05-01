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
    private long lastDraw;

    PlotState(Plot<Double, Double> plot, INDArray labels) {
        this.plot = plot;
        this.labels = labels;
        this.lastDraw = System.currentTimeMillis();
    }

    @Override
    public void begin(INDArray t, INDArray y0) {
        plotXY(y0);
    }

    @Override
    public void step(SolverState solverState, INDArray step, INDArray error) {
        plotXY(solverState.getCurrentState());
    }

    @Override
    public void done() {

    }

    private void plotXY(INDArray xyPoints) {
        final long procdelay = System.currentTimeMillis() - lastDraw;

        if(procdelay < 100) {
            try {
                Thread.sleep(100 - procdelay);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        final String posLab = "g(x) = 1";
        final String negLab = "g(x) = -1";
        plot.clearData(posLab);
        plot.clearData(negLab);
        plot.createSeries(posLab).scatter().set(Color.blue);
        plot.createSeries(negLab).scatter().set(Color.red);

        final double[] x;
        final double[] y;
        if (xyPoints.size(1) == 1) {
            y = xyPoints.toDoubleVector();
            x = new double[y.length]; // Init to zeros
        } else {
            x = xyPoints.getColumn(0).toDoubleVector();
            y = xyPoints.getColumn(1).toDoubleVector();
        }
        final int[] labs = labels.toIntVector();

        final Map<Integer, Pair<List<Double>, List<Double>>> data = new HashMap<>();
        data.put(1, new Pair<>(new ArrayList<>(), new ArrayList<>()));
        data.put(-1, new Pair<>(new ArrayList<>(), new ArrayList<>()));

        for (int i = 0; i < xyPoints.size(0); i++) {
            data.get(labs[i]).getFirst().add(x[i]);
            data.get(labs[i]).getSecond().add(y[i]);
        }

        plot.plotData(posLab, data.get(1).getFirst(), data.get(1).getSecond());
        plot.plotData(negLab, data.get(-1).getFirst(), data.get(-1).getSecond());

        lastDraw = System.currentTimeMillis();
    }

    static void plotXY(INDArray xyPoints, INDArray labels, Plot<Double, Double> plot) {
        new PlotState(plot, labels).plotXY(xyPoints);
    }
}
