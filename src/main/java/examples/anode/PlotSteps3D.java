package examples.anode;

import ode.solve.api.StepListener;
import ode.solve.impl.util.SolverState;
import org.jzy3d.colors.Color;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Triple;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Plots the state of each solver step
 *
 * @author Christian Skarby
 */
public class PlotSteps3D implements StepListener {
    private final Plot3D plot;
    private final INDArray labels;
    private long lastDraw;

    PlotSteps3D(Plot3D plot, INDArray labels) {
        this.plot = plot;
        this.labels = labels;
        this.lastDraw = System.currentTimeMillis();
    }

    @Override
    public void begin(INDArray t, INDArray y0) {
        plotXYZ(y0);
    }

    @Override
    public void step(SolverState solverState, INDArray step, INDArray error) {
        plotXYZ(solverState.getCurrentState());
    }

    @Override
    public void done() {

    }

    private void plotXYZ(INDArray xyzPoints) {
        final long procdelay = System.currentTimeMillis() - lastDraw;

        if(procdelay < 100) {
            try {
                Thread.sleep(100 - procdelay);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        final ScatterPlot3D.Series3D negLab = plot.series("g(x) = -1");
        final ScatterPlot3D.Series3D posLab = plot.series("g(x) = 1");
        negLab.clear().color(Color.BLUE).size(4);
        posLab.clear().color(Color.RED).size(4);

        final double[] x;
        final double[] y;
        final double[] z;
        if (xyzPoints.size(1) == 1) {
            y = xyzPoints.toDoubleVector();
            x = new double[y.length]; // Init to zeros
            z = new double[y.length];
        } else if(xyzPoints.size(1) == 2) {
            x = xyzPoints.getColumn(0).toDoubleVector();
            y = xyzPoints.getColumn(1).toDoubleVector();
            z = new double[y.length];
        }  else {
            x = xyzPoints.getColumn(0).toDoubleVector();
            y = xyzPoints.getColumn(1).toDoubleVector();
            z = xyzPoints.getColumn(2).toDoubleVector();
        }
        final int[] labs = labels.toIntVector();

        final Map<Integer, Triple<List<Double>, List<Double>, List<Double>>> data = new HashMap<>();
        data.put(1, new Triple<>(new ArrayList<>(), new ArrayList<>(), new ArrayList<>()));
        data.put(-1, new Triple<>(new ArrayList<>(), new ArrayList<>(), new ArrayList<>()));

        for (int i = 0; i < xyzPoints.size(0); i++) {
            data.get(labs[i]).getFirst().add(x[i]);
            data.get(labs[i]).getSecond().add(y[i]);
            data.get(labs[i]).getThird().add(z[i]);
        }

        posLab.plot(data.get(1).getFirst(), data.get(1).getSecond(), data.get(1).getThird());
        negLab.plot(data.get(-1).getFirst(), data.get(-1).getSecond(), data.get(-1).getThird());

        plot.fit();
        lastDraw = System.currentTimeMillis();
    }

    static void plotXYZ(INDArray xyzPoints, INDArray labels, Plot3D plot) {
        new PlotSteps3D(plot, labels).plotXYZ(xyzPoints);
    }

}
