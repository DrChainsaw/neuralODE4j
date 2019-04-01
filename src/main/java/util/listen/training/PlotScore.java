package util.listen.training;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
import util.plot.Plot;
import util.plot.RealTimePlot;

/**
 * Plot score over time
 *
 * @author Christian Skarby
 */
public class PlotScore extends BaseTrainingListener {

    private final Plot<Integer, Double> plot;

    public PlotScore() {
        this(new RealTimePlot<>("Score", ""));
    }

    public PlotScore(Plot<Integer, Double> plot) {
        this.plot = plot;
    }

    @Override
    public void iterationDone(Model model, int iteration, int epoch) {
        plot.plotData("Score", iteration, model.score());
    }
}
