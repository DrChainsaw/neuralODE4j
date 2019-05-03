package examples.anode;

import org.jzy3d.colors.Color;

import java.util.List;

public class NoPlot3D implements Plot3D {

    private static class NoSeries3D implements Series3D {

        @Override
        public Series3D plot(List<Double> x, List<Double> y, List<Double> z) {
            return this;
        }

        @Override
        public Series3D color(Color color) {
            return this;
        }

        @Override
        public Series3D size(float size) {
            return this;
        }

        @Override
        public Series3D clear() {
            return this;
        }
    }

    @Override
    public Series3D series(String label) {
        return new NoSeries3D();
    }

    @Override
    public void savePicture(String suffix) {
        // Ignore
    }

    @Override
    public void fit() {
        // Ignore
    }
}
