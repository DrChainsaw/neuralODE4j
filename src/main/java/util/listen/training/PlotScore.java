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
    private final double alpha;

    private double avgScore = Double.NaN;

    public PlotScore() {
        this(new RealTimePlot<>("Score", ""));
    }

    public PlotScore(Plot<Integer, Double> plot) {
        this(plot, 1);
    }

    public PlotScore(Plot<Integer, Double> plot, double alpha) {
        this.plot = plot;
        this.alpha = alpha;
    }


    @Override
    public void iterationDone(Model model, int iteration, int epoch) {
        if(Double.isNaN(avgScore)) {
            avgScore = model.score();
        } else {
            avgScore = alpha * model.score() + (1 - alpha) * avgScore;
        }

        plot.plotData("Score" + (alpha == 1.0 ? "" : "_"+alpha), iteration, avgScore);
    }
}
