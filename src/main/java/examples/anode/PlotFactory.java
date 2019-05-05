package examples.anode;

import com.beust.jcommander.Parameter;
import util.plot.NoPlot;
import util.plot.Plot;
import util.plot.RealTimePlot;

/**
 * Creates plots for the anode examples. Plotting can be turned off, e.g. when testing
 *
 * @author Christian Skarby
 */
class PlotFactory {

    @Parameter(names = {"-plotsOff"}, description = "Set to disable plotting")
    private boolean plotsOff = false;

    /**
     * Create a new plot with the given title
     * @param title title of the plot
     * @param plotDir Place to store plots
     * @return a new Plot instance
     */
    <X extends Number, Y extends  Number> Plot<X, Y> plot2D(String title, String plotDir) {
        if(plotsOff) {
            return new NoPlot<>();
        }
        return new RealTimePlot<>(title, plotDir);
    }

    /**
     * Create a new plot with the given title
     * @param title title of the plot
     * @param plotDir Place to store plots
     * @return a new Plot instance
     */
    Plot3D plot3D(String title, String plotDir) {
        if(plotsOff) {
            return new NoPlot3D();
        }

        return new ScatterPlot3D(title, plotDir);
    }
}
