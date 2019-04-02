package util.listen.training;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.factory.Nd4j;
import util.plot.Plot;
import util.plot.RealTimePlot;

/**
 * Plots size of workspaces per iteration.
 *
 * @author Christian Skarby
 */
public class PlotMem extends BaseTrainingListener {

    private final Plot<Integer, Long> memPlot;

    public PlotMem() {
        this(new RealTimePlot<>("Size per workspace", ""));
    }

    public PlotMem(Plot<Integer, Long> plot) {
        memPlot = plot;
    }

    @Override
    public void iterationDone(Model model, int iteration, int epoch) {
        for (MemoryWorkspace ws : Nd4j.getWorkspaceManager().getAllWorkspacesForCurrentThread()) {
            memPlot.plotData(ws.getId(), iteration, ws.getCurrentSize());
        }
    }
}
